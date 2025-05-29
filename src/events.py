import time


class HpActionController:
    def __init__(
        self,
        send_command_callback,
        spoil_key="F2",
        no_target_command=None,
        dead_target_command=None,
        alive_target_command=None,
        spoil_enabled=True,
        cooldown_sec=0.5,
    ):
        """
        Контроллер для управления действиями по HP цели с поддержкой спойла и свипа.

        :param send_command_callback: функция для отправки команды (строка, например 'F2')
        :param spoil_key: команда для спойла (например, 'F2')
        :param no_target_command: команда при отсутствии цели
        :param dead_target_command: команда при мёртвой цели
        :param alive_target_command: команда при живой цели (атака)
        :param spoil_enabled: включён ли спойл
        :param cooldown_sec: минимальный интервал между отправками команд
        """
        self.send_command_callback = send_command_callback

        self.spoil_key = spoil_key
        self.no_target_command = no_target_command
        self.dead_target_command = dead_target_command
        self.alive_target_command = alive_target_command

        self.spoil_enabled = spoil_enabled
        self.cooldown_sec = cooldown_sec

        self.current_state = None  # "no_target", "dead_target", "alive_target", None
        self.last_command = None
        self.last_command_time = 0

        self.spoil_active = None
        self.waiting_for_sweep = False

        self.enabled = True  # общий флаг включения контроллера

    def set_spoil_key(self, key: str):
        print(f"[HpActionController] Установлен spoil_key: {key}")
        self.spoil_key = key.strip()

    def set_no_target_command(self, cmd: str):
        print(f"[HpActionController] Установлена команда no_target: {cmd}")
        self.no_target_command = cmd.strip() if cmd else None

    def set_dead_target_command(self, cmd: str):
        print(f"[HpActionController] Установлена команда dead_target: {cmd}")
        self.dead_target_command = cmd.strip() if cmd else None

    def set_alive_target_command(self, cmd: str):
        print(f"[HpActionController] Установлена команда alive_target: {cmd}")
        self.alive_target_command = cmd.strip() if cmd else None

    def set_sweep_key(self, key: str):
        print(f"[HpActionController] Установлен sweep_key: {key}")
        self.sweep_key = key.strip() if key else None

    def set_cooldown(self, cooldown: float):
        print(f"[HpActionController] Установлен cooldown: {cooldown}")
        self.cooldown_sec = cooldown

    def set_spoil_enabled(self, enabled: bool):
        print(f"[HpActionController] Установлен spoil_enabled: {enabled}")
        self.spoil_enabled = enabled

    def set_spoil_state(self, is_spoiled: bool, can_sweep: bool):
        self.spoil_active = is_spoiled
        self.waiting_for_sweep = can_sweep
        print(
            f"[HpActionController] set_spoil_state: spoil_active={self.spoil_active}, waiting_for_sweep={self.waiting_for_sweep}"
        )

    def update(self, target_state: str, hp_percent: float):
        print(
            f"[HpActionController] update start: spoil_active={self.spoil_active}, current_state={self.current_state}, target_state={target_state}"
        )

        if not self.enabled:
            print("[HpActionController] update: обработка выключена, ничего не делаем")
            return

        now = time.time()

        if self.spoil_enabled:
            if (
                self.current_state != target_state
            ) and target_state == "alive_target":  # цель изменилась
                print("[HpActionController] Новая цель — сброс состояния спойла")
                self.spoil_active = False
                self.waiting_for_sweep = False

        """
        Обновить состояние цели и отправить команду при необходимости.
        Также управляет логикой спойла и свипа.

        :param target_state: "no_target", "dead_target", "alive_target"
        :param hp_percent: процент HP (float)
        """
        print(
            f"[HpActionController] update: target_state={target_state}, hp_percent={hp_percent}, enabled={self.enabled}"
        )

        if not self.enabled:
            print("[HpActionController] update: обработка выключена, ничего не делаем")
            return

        now = time.time()

        # Сброс спойла при смене цели на живую
        if self.spoil_enabled:
            if (
                self.current_state in [None, "no_target", "dead_target"]
                and target_state == "alive_target"
                and not self.spoil_active  # Добавлено условие
            ):
                print("[HpActionController] Новая цель — сброс состояния спойла")
                self.spoil_active = False
                self.waiting_for_sweep = False

            # Если цель умерла и спойл активен и можно свипать, отправляем свип
            if (
                target_state == "dead_target"
                and self.spoil_active
                and self.waiting_for_sweep
            ):
                self.try_sweep()
                self.waiting_for_sweep = False

        cmd_map = {
            "no_target": self.no_target_command,
            "dead_target": self.dead_target_command,
            "alive_target": None,  # обработаем отдельно
        }

        # Отправка команд для no_target и dead_target
        if target_state in ["no_target", "dead_target"]:
            cmd = cmd_map.get(target_state)
            if cmd and (
                self.last_command != cmd
                or now - self.last_command_time > self.cooldown_sec
            ):
                print(f"[HpActionController] Отправляю команду {target_state}: {cmd}")
                self.send_command_callback(cmd)
                self.last_command = cmd
                self.last_command_time = now

        # Обработка alive_target с учётом спойла
        elif target_state == "alive_target":
            if self.spoil_enabled:
                if not self.spoil_active:
                    print(
                        "[HpActionController] Спойл не активен, пытаемся применить спойл"
                    )
                    self.try_spoil()
                else:
                    if self.alive_target_command and (
                        self.last_command != self.alive_target_command
                        or now - self.last_command_time > self.cooldown_sec
                    ):
                        print("[HpActionController] Спойл активен — атакуем цель")
                        self.send_command_callback(self.alive_target_command)
                        self.last_command = self.alive_target_command
                        self.last_command_time = now
            else:
                if self.alive_target_command and (
                    self.last_command != self.alive_target_command
                    or now - self.last_command_time > self.cooldown_sec
                ):
                    print(
                        f"[HpActionController] Отправляю команду alive_target: {self.alive_target_command}"
                    )
                    self.send_command_callback(self.alive_target_command)
                    self.last_command = self.alive_target_command
                    self.last_command_time = now
        else:
            print(f"[HpActionController] Неизвестное состояние: {target_state}")

        self.current_state = target_state

    def try_spoil(self):
        if self.spoil_key and not self.spoil_active:
            print(f"[HpActionController] Отправляю Spoil: {self.spoil_key}")
            self.send_command_callback(self.spoil_key)
        else:
            print("[HpActionController] Спойл уже активен или ключ не задан")

    def try_sweep(self):
        if hasattr(self, "sweep_key") and self.sweep_key:
            print(f"[HpActionController] Отправляю Sweep: {self.sweep_key}")
            self.send_command_callback(self.sweep_key)
        else:
            print("[HpActionController] Ключ свипа не задан")

    def stop(self):
        self.enabled = False
        self.spoil_active = False
        self.waiting_for_sweep = False
        self.current_state = None
        self.last_command = None
        self.last_command_time = 0
        print("[HpActionController] Контроллер остановлен")

    def start(self):
        self.enabled = True
        print("[HpActionController] Контроллер запущен")
