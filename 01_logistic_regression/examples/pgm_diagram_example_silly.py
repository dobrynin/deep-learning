import daft

ASPECT = 2.5

def run():
    pgm = daft.PGM([7, 3])

    offer_node = pgm.add_node(daft.Node(
        'offer?',
        'offer?',
        y = 2.5,
        x = (ASPECT / 2 + 0.1) * 1,
        aspect = ASPECT,
    ))
    money_node = pgm.add_node(daft.Node(
        'money?',
        'money?',
        y = 2.5,
        x = (ASPECT / 2 + 0.1) * 4,
        aspect = ASPECT,
    ))
    pgm.add_edge('offer?', 'money?')

    spam_node = pgm.add_node(daft.Node(
        'spam?',
        'spam?',
        y = 1.5,
        x = (ASPECT / 2 + 0.1) * 2.5,
        aspect = ASPECT
    ))
    pgm.add_edge('offer?', 'spam?', directed = True)
    pgm.add_edge('money?', 'spam?', directed = True)

    words = [
        ('investment?', 1),
        ('consultation?', 4),
    ]

    for word, position in words:
        node = daft.Node(
            word,
            word,
            y = 0.5,
            x = (ASPECT / 2 + 0.1) * position,
            aspect = ASPECT
        )
        pgm.add_node(node)
        pgm.add_edge('spam?', word, directed = True)

    pgm.render()
