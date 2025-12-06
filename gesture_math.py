class GestureMathEngine:
    """
    Simple state machine for gesture-based arithmetic:
    IDLE -> OPERAND1 -> OPERATOR -> OPERAND2 -> RESULT -> IDLE
    """

    def __init__(self):
        self.reset()
        self.last_gesture = None
        self.cooldown = 0

    def reset(self):
        self.state = "IDLE"
        self.operand1 = None
        self.operator = None   # "add" | "sub" | "mul"
        self.operand2 = None
        self.result = None
        self.cooldown = 0

    def stable_number(self, left, right):
        """Choose a number from left / right finger counts."""
        if left is None and right is None:
            return None
        return max(left or 0, right or 0)

    def update(self, left, right, gesture=None):
        """
        left, right: finger counts or None
        gesture: "add" | "sub" | "mul" | "eval" | None
        """
        # Global gesture cooldown (except while showing RESULT)
        if self.state != "RESULT" and self.cooldown > 0:
            self.cooldown -= 1
            gesture = None

        number = self.stable_number(left, right)

        if self.state == "IDLE":
            if number is not None:
                self.operand1 = number
                self.state = "OPERAND1"

        elif self.state == "OPERAND1":
            if gesture in ("add", "sub", "mul"):
                self.operator = gesture
                self.cooldown = 15  # small pause after operator
                self.state = "OPERATOR"

        elif self.state == "OPERATOR":
            if number is not None:
                self.operand2 = number
                self.state = "OPERAND2"

        elif self.state == "OPERAND2":
            if gesture == "eval":
                self.compute()
                self.cooldown = 30  # show result for some frames
                self.state = "RESULT"

        elif self.state == "RESULT":
            # Once cooldown expires, reset back to IDLE
            if self.cooldown > 0:
                self.cooldown -= 1
            else:
                self.reset()

        self.last_gesture = gesture
        return self.state

    def compute(self):
        if (
            self.operator
            and self.operand1 is not None
            and self.operand2 is not None
        ):
            if self.operator == "add":
                self.result = self.operand1 + self.operand2
            elif self.operator == "sub":
                self.result = self.operand1 - self.operand2
            elif self.operator == "mul":
                self.result = self.operand1 * self.operand2
        else:
            self.result = "?"

    def get_operator_symbol(self):
        return {"add": "+", "sub": "-", "mul": "*"}.get(self.operator, "?")

    def get_display(self):
        """User-facing text for HUD."""
        if self.state == "IDLE":
            return "Show a number to start"
        if self.state == "OPERAND1":
            return f"{self.operand1} | Show +  -  *"
        if self.state == "OPERATOR":
            return f"{self.operand1} {self.get_operator_symbol()} ?"
        if self.state == "OPERAND2":
            return f"{self.operand1} {self.get_operator_symbol()} {self.operand2} | Clap!"
        if self.state == "RESULT":
            return f"{self.operand1} {self.get_operator_symbol()} {self.operand2} = {self.result}"
        return "..."
