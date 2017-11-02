import daft

def run():
    pgm = daft.PGM([4, 4])

    nodes = [
        (1, 3, 1),
        (2, 2, 1),
        (3, 1, 2),
        (4, 1, 3),
    ]

    for node in nodes:
        (num, y, x) = node
        pgm.add_node(daft.Node(
            f"X_{num}",
            f"X_{num}",
            x = x,
            y = y,
        ))
        for prev_num in range(1, num):
            pgm.add_edge(
                f"X_{prev_num}",
                f"X_{num}",
                directed = True,
            )

    pgm.render()
