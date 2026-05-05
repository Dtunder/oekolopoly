"""
    New (recommended) version to get the value for each box.

    A box is an element on the Ökolopoly game board that carries a number depending on the value of one specific
    observation wheel (the argument of ``get_box*``).
"""
class GetBoxes:
    """
        This class allows to get the complete list for each box via ``GetBoxes.box*`` (needed by
        ``plot_sector_diag.py``).

        ``GetBoxes.get_box*`` relies on these lists and is therefore faster than in ``get_boxes.py``.
    """
    # Box 1 Sanitation-Environment 0-5 (0 = SANITATION, 5 = ENVIRONMENT, see oeko_env.py:29)
    #        0                   5
    box1 = [ 0,  0, -1, -1, -1, -1, -1, -1, -2, -2,
            -2, -2, -2, -3, -3, -3, -3, -3, -4, -4,
            -4, -5, -5, -5, -6, -6, -7, -7, -8, -9,
            ]

    # Box 2 Sanitation-Sanitation 0-0
    #        0                   5
    box2 = [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0, -1, -1, -2, -2, -3, -4, -5, -6,
            ]

    # Box 3 Production-Production 1-1
    #       0              5
    box3 = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 2, 2, 2, 2, 2, 0, 0, 0,
            ]

    # Box 4 Production-Environment 1-5
    #       0              5
    box4 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2,
            2, 2, 2, 3, 3, 4, 4, 5, 5, 6,
            6, 7, 7, 8, 9,10,12,14,18,22,
            ]

    # Box 5 Environment-Environment 5-5
    #       0                   5
    box5 = [0,  0,  0,  0, -1, -1, -1, -1, -1, -1,
           -1, -1, -1, -1, -2, -2, -2, -2, -2, -2,
           -3, -3, -3, -3, -4, -4, -3, -2, -1,  0,
            ]

    # Box 6 Environment-Quality of Life 5-3
    #        0                   5
    box6 = [ 0,  0,  0,  0,  0,  0,  0,  0, -1, -1,
            -1, -2, -2, -2, -2, -3, -3, -3, -4, -4,
            -5, -5, -6, -7, -8,-10,-12,-14,-18,-25,
            ]

    # Box 7 Education-Education 2-2
    #       0                   5
    box7 = [0,  0,  0, -1, -1, -1,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  1,  1,  1,  1,  1,
            1,  2,  2,  2,  1,  1,  1,  1,  1,  0,
            ]

    # Box 8 Education-Quality of Life 2-3
    #        0                   5
    box8 = [-2, -2, -2, -2, -2, -2, -1, -1, -1, -1,
             0,  1,  1,  1,  2,  2,  2,  2,  3,  3,
             3,  3,  4,  4,  4,  4,  5,  5,  6,  6,
            ]

    # Box 9 Education-Population Growth 2-4
    #        0                   5
    box9 = [ 1,  1,  0,  0,  0,  0,  0,  0,  0,  0,
             1,  1,  1,  1,  2,  2,  2,  2,  2,  3,
             3,  3,  3,  3,  4,  4,  4,  4,  5,  5,
            ]

    # Box 10 Quality of Life-Quality of Life 3-3
    #         0                   5
    box10 = [ 0,  0,  1,  1,  0,  0,  0,  0,  0,  0,
              1,  1,  1,  2,  1,  1,  0,  0, -1, -1,
             -1, -1, -1, -2, -2, -2, -1, -1, -1,  0,
             ]

    # Box 11 Quality of Life-Population Growth 3-4
    #         0                   5
    box11 = [ 0,-15, -8, -6, -4, -3, -2, -1,  0,  1,
              2,  2,  2,  1,  1,  1,  1,  1,  1,  1,
              1,  0,  0,  0,  0,  0,  0,  0,  0,  0,
             ]

    # Box 12 Quality of Life-Politics 3-7
    #         0                   5
    box12 = [ 0,-10, -8, -6, -3, -2, -1, -1, -1, -1,
              0,  0,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  2,  2,  2,  3,  3,  3,  4,  4,  5,
             ]

    # Box 13 Population Growth-Population 4-6
    #         0                   5
    box13 = [ 0, -4, -4, -3, -3, -3, -2, -2, -2, -2,
             -1, -1, -1, -1, -1,  0,  1,  1,  1,  1,
              1,  2,  2,  2,  2,  2,  3,  3,  3,  3,
             ]

    # Box 14 Population-Quality of Life 6-3
    #         0                   5
    box14 = [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0, -1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1, -2, -2, -2, -2,
             -2, -3, -3, -3, -3, -3, -3, -3, -3, -4,
             -4, -4, -4, -5, -5, -6, -7, -8,-10,-10,
             ]

    # Box A (=15) Population-Action points 6-9
    #         0                   5
    boxA  = [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
              1,  1,  1,  1,  1,  1,  1,  2,  2,  2,
              2,  2,  3,  3,  3,  3,  4,  4,  4,  4,
              5,  5,  5,  5,  6,  6,  6,  6,  7,  7,
              7,  7,  8,  8,  8,  8,  9,  9,  9,  9,
             ]

    # Box B (=16) Politics-Action points 7-9
    #         0                   5                 (argument has +10 added)
    boxB  = [-5, -2, -1, -1, -1, -1, -1, -1,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  2,  2,  2,  2,  2,  2,  2,  2,
              2,  3,  3,  3,  3,  3,  3,  3,  3,  3,
             ]

    # Box C Production-Action points 1-9
    #         0                   5
    boxC  = [ 0, -4, -3, -2, -1,  0,  2,  2,  3,  3,
              3,  4,  4,  4,  5,  5,  5,  6,  6,  6,
              7,  7,  8,  8,  9,  9, 10, 10, 11, 11,
             ]

    # Box D Quality of Life-Action points 3-9
    #         0                   5
    boxD  = [-6, -6, -4, -2,  0,  0,  1,  1,  1,  2,
              2,  2,  2,  2,  2,  2,  2,  2,  3,  3,
              3,  3,  3,  4,  4,  4,  4,  5,  5,  5,
             ]

    # Box V: Multiplier for boxA as a function of production 1
    #         0                   5
    boxV  = [-4, -4, -3, -2, -2, -1, -1,  0,  0,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
             ]

    # Box W: Multiplier for box13 as a function of Population 6
    #         0                   5
    boxW  = [ 0,  0,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  2,  2,  2,  2,  2,
              2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
              2,  2,  2,  2,  2,  2,  3,  3,  3,  3,
              3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
             ]

    def get_box1(sanitation: int):
        return GetBoxes.box1[sanitation]

    def get_box2(sanitation: int):
        return GetBoxes.box2[sanitation]

    def get_box3(production: int):
        return GetBoxes.box3[production]

    def get_box4(production: int):
        return GetBoxes.box4[production]

    def get_box5(environment: int):
        return GetBoxes.box5[environment]

    def get_box6(environment: int):
        return GetBoxes.box6[environment]

    def get_box7(education: int):
        return GetBoxes.box7[education]

    def get_box8(education: int):
        return GetBoxes.box8[education]

    def get_box9(education: int, extra_points):
        return GetBoxes.box9[education] if education < 21 else extra_points

    def get_box10(quality_of_life: int):
        return GetBoxes.box10[quality_of_life]

    def get_box11(quality_of_life: int):
        return GetBoxes.box11[quality_of_life]

    def get_box12(quality_of_life: int):
        return GetBoxes.box12[quality_of_life]

    def get_box13(population_growth: int):
        return GetBoxes.box13[population_growth]

    def get_box14(population: int):
        return GetBoxes.box14[population]

    def get_boxA(population: int):
        return GetBoxes.boxA[population]

    def get_boxB(politics: int):
        return GetBoxes.boxB[politics + 10]

    def get_boxC(production: int):
        return GetBoxes.boxC[production]

    def get_boxD(quality_of_life: int):
        return GetBoxes.boxD[quality_of_life]

    def get_boxV(production: int):
        return GetBoxes.boxV[production]

    def get_boxW(population: int):
        return GetBoxes.boxW[population]
