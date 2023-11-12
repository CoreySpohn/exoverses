class Star:
    """
    The star of a system
    """

    def __init__(self, star_dict):
        self.spectral_type = star_dict["spectral_type"]
        self.dist = star_dict["dist"]
        self.name = star_dict["name"]
        self.mass = star_dict["mass"]

    def __repr__(self):
        return f"{type(self).__name__} object\n{self.name}"
