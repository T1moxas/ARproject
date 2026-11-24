from contextlib import ExitStack
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from depth_appearance import STYLES
from depth_http_preview import ProjectionConflict
from projector_output import Display, ProjectionControl, ProjectorOutput
from surface_geometry import FloorPlane, HeightRenderer, ProjectorModel


class SharedGeometryTests(unittest.TestCase):
    def test_fifteen_millimeter_floor_band_keeps_small_noise_in_floor_color(self):
        renderer = HeightRenderer(FloorPlane([0, 0, -1, 1.5], "calibration"), np.eye(3), noise=.015)
        depth = np.float32([[1.5, 1.49, 1.486, 1.482, 1.518]])
        colors = renderer.colorize(depth)[0]
        np.testing.assert_array_equal(colors[1], colors[0])
        np.testing.assert_array_equal(colors[2], colors[0])
        self.assertFalse(np.array_equal(colors[3], colors[0]))
        self.assertFalse(colors[4].any())

    def test_both_layers_clip_and_occlude_in_the_same_geometry(self):
        intrinsic = np.float64([[100, 0, 60], [0, 100, 40], [0, 0, 1]])
        matrix = np.float64([[200, 0, 120, -20], [0, 200, 80, 0], [0, 0, 1, 0]])
        model = ProjectorModel(matrix, (240, 160), "calibration")
        depth = np.ones((80, 120), np.float32)
        colors = np.full((80, 120, 3), (180, 53, 7), np.uint8)
        web = np.full((80, 120, 4), (81, 69, 60, 165), np.uint8)
        depth[30:50, 80:90] = .5
        colors[30:50, 80:90] = 0
        web[30:50, 80:90] = (120, 111, 107, 197)
        depth[15:25, 55:65] = np.nan
        rectangle = (10, 5, 230, 155)

        projector, site = model.render(colors, depth, intrinsic, rectangle, web=web)
        self.assertEqual(projector.shape, (160, 240, 3))
        self.assertEqual(site.shape, (160, 240, 4))
        np.testing.assert_array_equal(projector, model.render(colors, depth, intrinsic, rectangle))
        np.testing.assert_array_equal(projector[80, 130], (0, 0, 0))
        np.testing.assert_array_equal(site[80, 130], (120, 111, 107, 197))
        np.testing.assert_array_equal(projector[80, 30], (180, 53, 7))
        np.testing.assert_array_equal(site[80, 30], (81, 69, 60, 165))
        np.testing.assert_array_equal(projector[40, 100], (0, 0, 0))
        np.testing.assert_array_equal(site[40, 100], (0, 0, 0, 0))
        for layer in (projector, site):
            self.assertFalse(layer[:5].any())
            self.assertFalse(layer[155:].any())
            self.assertFalse(layer[:, :10].any())
            self.assertFalse(layer[:, 230:].any())
        for web_size in ((105, 71), (96, 64), (1, 1)):
            with self.subTest(web_size=web_size):
                native, compact = model.render(colors, depth, intrinsic, rectangle, web=web, web_size=web_size)
                np.testing.assert_array_equal(native, projector)
                np.testing.assert_array_equal(compact,
                                              cv2.resize(site, web_size, interpolation=cv2.INTER_NEAREST))

    def test_compact_web_projection_preserves_empty_raster_dimensions(self):
        model = ProjectorModel(np.column_stack((np.eye(3), np.zeros(3))), (40, 32), "calibration")
        depth = np.full((32, 40), np.nan, np.float32)
        projector, site = model.render(np.zeros((32, 40, 3), np.uint8), depth, np.eye(3), (0, 0, 40, 32),
                                       web=np.zeros((32, 40, 4), np.uint8), web_size=(20, 16))
        self.assertEqual(projector.shape, (32, 40, 3))
        self.assertEqual(site.shape, (16, 20, 4))
        self.assertFalse(projector.any())
        self.assertFalse(site.any())


class FakeDisplays:
    projector = r"\.DISPLAY2"

    def __init__(self):
        self.active = (1920, 1200)
        self.switches = []

    def displays(self):
        return [
            Display(r"\.DISPLAY1", "Оператор", 0, 0, 1920, 1080, True, ((1920, 1080),)),
            Display(self.projector, "Проектор", -1920, 40, 1920, 1200, False,
                    ((1280, 800), (1920, 1080), (1920, 1200), (2560, 1600))),
        ]

    def mode(self, display):
        width, height = self.active if display == self.projector else (1920, 1080)
        return SimpleNamespace(width=width, height=height, position=SimpleNamespace(x=-1920, y=40))

    def select_mode(self, display, size):
        previous = self.mode(display)
        self.switches.append((display, size))
        self.active = size
        return previous

    def restore(self, display, previous):
        self.active = (previous.width, previous.height)
        self.switches.append((display, self.active))


class PhysicalProjectionTests(unittest.TestCase):
    def setUp(self):
        self.displays = FakeDisplays()
        self.output = ProjectorOutput("Projector", (1280, 800), -100, 50, True,
                                      backend=self.displays)

    def test_uses_active_display_and_filters_unsupported_or_anisotropic_modes(self):
        self.assertEqual(self.output.initial, {
            "display": self.displays.projector, "width": 1920, "height": 1200, "fullscreen": True,
        })
        modes = self.output.caller_displays()
        self.assertEqual(modes[0]["modes"], [])
        self.assertEqual([(mode["width"], mode["height"]) for mode in modes[1]["modes"]],
                         [(1280, 800), (1920, 1200), (2560, 1600)])
        for width, height in [(1920, 1080), (1600, 1000)]:
            with self.subTest(size=(width, height)), self.assertRaises(ValueError):
                self.output.validate(dict(self.output.initial, width=width, height=height))
        self.assertEqual(self.displays.switches, [])

    def test_supported_fullscreen_switch_keeps_every_corner_and_restores_display(self):
        initial = self.output.initial
        requested = dict(initial, width=2560, height=1600)
        frame = np.empty((800, 1280, 3), np.uint8)
        frame[:400, :640] = (10, 20, 30)
        frame[:400, 640:] = (50, 60, 70)
        frame[400:, :640] = (90, 100, 110)
        frame[400:, 640:] = (130, 140, 150)
        with ExitStack() as stack:
            window = {name: stack.enter_context(patch.object(cv2, name)) for name in
                      ("resizeWindow", "moveWindow", "waitKey", "setWindowProperty", "imshow")}
            self.output.configure(initial)
            self.assertEqual(self.displays.switches, [])
            self.output.configure(requested)
            window["resizeWindow"].assert_called_with("Projector", 2560, 1600)
            window["moveWindow"].assert_called_with("Projector", -1920, 40)
            window["setWindowProperty"].assert_called_with(
                "Projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            projected = self.output.present(frame)
            self.assertIs(window["imshow"].call_args.args[1], projected)
            self.output.restore_mode()
        self.assertEqual(projected.shape, (1600, 2560, 3))
        np.testing.assert_array_equal(projected[[0, 0, -1, -1], [0, -1, 0, -1]],
                                      frame[[0, 0, -1, -1], [0, -1, 0, -1]])
        self.assertEqual(self.displays.switches,
                         [(self.displays.projector, (2560, 1600)), (self.displays.projector, (1920, 1200))])

    def test_keyboard_cycles_apply_on_calling_thread_and_advance_settings_revision(self):
        control = ProjectionControl(self.output, "smooth", "surface", STYLES, 15.0)
        changes = []
        thread_id = threading.get_ident()

        def apply(settings, old):
            changes.append((threading.get_ident(), settings["appearance"], settings["fill"],
                            old["appearance"], old["fill"]))

        old = control.state()
        control.cycle("appearance", apply)
        control.cycle("fill", apply)
        state = control.state()
        self.assertEqual(changes,
                         [(thread_id, "raw", "surface", "smooth", "surface"),
                          (thread_id, "raw", "conservative", "raw", "surface")])
        self.assertEqual(state["revision"], 2)
        self.assertEqual(state["calibration"]["noise_mm"], 15.0)
        with self.assertRaises(ProjectionConflict):
            control.change(dict(old["settings"], expected_revision=old["revision"]), apply)
        self.assertEqual(control.state(), state)
