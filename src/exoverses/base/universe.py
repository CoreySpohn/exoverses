class Universe:
    """
    The base class for universe. Keeps track of the planetary systems.
    """

    def __init__(self) -> None:
        pass

    def __repr__(self):
        str = f"{self.type} universe\n"
        str += f"{len(self.systems)} systems loaded"
        return str
