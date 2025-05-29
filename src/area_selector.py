import tkinter as tk


class AreaSelector(tk.Toplevel):
    def __init__(self, master, on_area_selected):
        super().__init__(master)
        self.on_area_selected = on_area_selected

        self.attributes("-fullscreen", True)
        self.attributes("-alpha", 0.3)
        self.configure(bg="black")

        self.canvas = tk.Canvas(self, cursor="cross", bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.start_x = None
        self.start_y = None
        self.selection_rect = None

        self.bind("<ButtonPress-1>", self.on_mouse_down)
        self.bind("<B1-Motion>", self.on_mouse_move)
        self.bind("<ButtonRelease-1>", self.on_mouse_up)

        print("[DEBUG] AreaSelector opened (fullscreen)")

    def on_mouse_down(self, event):
        self.start_x = self.winfo_pointerx()
        self.start_y = self.winfo_pointery()
        # Начинаем рисовать прямоугольник от текущей позиции мыши на canvas
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        self.selection_rect = self.canvas.create_rectangle(
            canvas_x, canvas_y, canvas_x, canvas_y, outline="red", width=2
        )
        print(
            f"[DEBUG] Mouse down at absolute ({self.start_x}, {self.start_y}), canvas ({canvas_x}, {canvas_y})"
        )

    def on_mouse_move(self, event):
        if self.selection_rect:
            # Координаты текущей позиции мыши на canvas
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            # Получаем координаты начальной точки в canvas
            start_canvas_x, start_canvas_y, _, _ = self.canvas.coords(
                self.selection_rect
            )
            # Обновляем координаты прямоугольника так, чтобы он рисовался правильно независимо от направления выделения
            x1 = min(start_canvas_x, canvas_x)
            y1 = min(start_canvas_y, canvas_y)
            x2 = max(start_canvas_x, canvas_x)
            y2 = max(start_canvas_y, canvas_y)
            self.canvas.coords(self.selection_rect, x1, y1, x2, y2)
            print(
                f"[DEBUG] Mouse move at canvas ({canvas_x}, {canvas_y}), rect coords: {(x1, y1, x2, y2)}"
            )

    def on_mouse_up(self, event):
        if self.selection_rect:
            end_x = self.winfo_pointerx()
            end_y = self.winfo_pointery()

            x1 = min(self.start_x, end_x)
            y1 = min(self.start_y, end_y)
            x2 = max(self.start_x, end_x)
            y2 = max(self.start_y, end_y)

            selected_area = (x1, y1, x2 - x1, y2 - y1)
            print(
                f"[DEBUG] Mouse up at absolute ({end_x}, {end_y}), selected area: {selected_area}"
            )
            self.destroy()
            if self.on_area_selected:
                self.on_area_selected(selected_area)


# Пример использования
if __name__ == "__main__":

    def on_area_selected(area):
        print("Выбранная область:", area)

    root = tk.Tk()
    root.withdraw()  # Скрыть основное окно
    selector = AreaSelector(root, on_area_selected)
    root.mainloop()
