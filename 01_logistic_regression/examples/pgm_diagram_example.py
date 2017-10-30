import daft

ASPECT = 2.5

def run():
    pgm = daft.PGM([7, 2])

    class_node = pgm.add_node(daft.Node(
        "spam?",
        "spam?",
        y = 1.5, x = (ASPECT / 2 + 0.1) * 2.5, aspect = ASPECT
    ))

    words = [
        "offer",
        "money",
        "investment",
        "consultation"
    ]

    for idx, word in enumerate(words):
        node = daft.Node(
            word, word, y = 0.5, x = (ASPECT / 2 + 0.1) * (idx + 1), aspect = ASPECT
        )
        pgm.add_node(node)
        pgm.add_edge("spam?", word, directed = True)

    pgm.render()
