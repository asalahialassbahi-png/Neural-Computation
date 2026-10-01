"""
Every way of driving the demo, turned into five events:

    "model+" / "model-"   next / previous network
    "view+"               next visual (network -> race -> equation -> neurons)
    "digit"               next MNIST digit
    "tour"                toggle the automatic tour

Sources (any or all may be connected):

  Arcade button       Pi GPIO27 (pin 13) to GND (pin 14):
                      short press = next digit, hold 1 s = next visual
  Mouse scroll wheel  salvaged from a broken mouse. A mechanical wheel is a
  (on GPIO)           3-pin quadrature encoder: outer pins A -> GPIO5 (pin 29),
                      B -> GPIO6 (pin 31), middle (common) -> GND (pin 30).
                      Turn = change network. The wheel's click switch:
                      one leg -> GPIO13 (pin 33), the other -> GND (pin 34);
                      click = next visual. (An optical wheel won't work this
                      way: use a KY-040 encoder, same three wires.)
  USB mouse           through a powered OTG hub (the Pi's only USB port feeds
                      the screen): wheel = network, left/middle = visual,
                      right = next digit
  Keyboard            <- -> network, up/down or V visual, SPACE digit,
                      1-5 pick a network, T tour, M mirror, F fullscreen, ESC quit

GPIO events arrive on gpiozero threads; they are queued and handled by the
render loop, so drawing never races with input.
"""
import queue
import time

GPIO_BUTTON, GPIO_WHEEL_A, GPIO_WHEEL_B, GPIO_WHEEL_CLICK = 27, 5, 6, 13
HOLD_S = 1.0


class Controls:
    def __init__(self, use_gpio=True):
        self.q = queue.Queue()
        self.last_input = time.time()
        self._gpio = []
        self._held = False
        if use_gpio:
            self._setup_gpio()

    def push(self, ev):
        self.last_input = time.time()
        self.q.put(ev)

    def _setup_gpio(self):
        try:
            from gpiozero import Button
            b = Button(GPIO_BUTTON, pull_up=True, bounce_time=0.05, hold_time=HOLD_S)
            b.when_held = self._on_held
            b.when_released = self._on_released
            self._gpio.append(b)
        except Exception as e:
            print("arcade button unavailable:", e)
        try:
            from gpiozero import RotaryEncoder
            enc = RotaryEncoder(GPIO_WHEEL_A, GPIO_WHEEL_B, max_steps=0, bounce_time=0.002)
            enc.when_rotated_clockwise = lambda: self.push("model+")
            enc.when_rotated_counter_clockwise = lambda: self.push("model-")
            self._gpio.append(enc)
        except Exception as e:
            print("scroll-wheel encoder unavailable:", e)
        try:
            from gpiozero import Button
            click = Button(GPIO_WHEEL_CLICK, pull_up=True, bounce_time=0.05)
            click.when_pressed = lambda: self.push("view+")
            self._gpio.append(click)
        except Exception as e:
            print("wheel click unavailable:", e)

    def _on_held(self):
        self._held = True
        self.push("view+")

    def _on_released(self):
        if not self._held:
            self.push("digit")
        self._held = False

    def handle_pygame(self, pg, ev):
        """Translate one pygame event; returns a window command or None."""
        if ev.type == pg.MOUSEWHEEL:
            self.push("model+" if ev.y < 0 else "model-")
        elif ev.type == pg.MOUSEBUTTONDOWN and ev.button in (1, 2):
            self.push("view+")
        elif ev.type == pg.MOUSEBUTTONDOWN and ev.button == 3:
            self.push("digit")
        elif ev.type == pg.KEYDOWN:
            k = ev.key
            if k in (pg.K_RIGHT,):
                self.push("model+")
            elif k in (pg.K_LEFT,):
                self.push("model-")
            elif k in (pg.K_UP, pg.K_DOWN, pg.K_v):
                self.push("view+")
            elif k == pg.K_SPACE:
                self.push("digit")
            elif k == pg.K_t:
                self.push("tour")
            elif pg.K_1 <= k <= pg.K_9:
                self.push(f"pick{k - pg.K_1}")
            elif k == pg.K_m:
                return "mirror"
            elif k == pg.K_f:
                return "fullscreen"
            elif k == pg.K_ESCAPE:
                return "quit"
        return None

    def events(self):
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out
