import math
import unittest

from model import BASE, EDIT, RATE, examples, forward, gradients, train


class Word2vecTests(unittest.TestCase):
    def test_examples_reverse_direction_and_do_not_cross_boundaries(self):
        vocab = ['bird', 'cat', 'eats', 'fish', 'the']
        data = ['the cat eats fish', 'bird']
        self.assertEqual(list(examples(data, vocab, 'cbow', 1))[1], ([4, 2], 1))
        self.assertIn(([1], 2), list(examples(data, vocab, 'skip', 1)))
        self.assertEqual(len(list(examples(data, vocab, 'skip', 1))), 6)
        self.assertIn(([1], 3), list(examples(data, vocab, 'skip', 2)))

    def test_forward_by_hand(self):
        h, z, p, loss = forward([[2, 0], [0, 2]], [[1, 0], [0, 0]], [0, 1], 0)
        self.assertEqual(h, [1, 1])
        self.assertEqual(z, [1, 0])
        self.assertAlmostEqual(p[0], 1 / (1 + math.exp(-1)))
        self.assertAlmostEqual(loss, -math.log(p[0]))

    def test_backprop_matches_finite_differences(self):
        E, U = [[.3, .4], [.2, -.8]], [[-.7, .3], [.4, -.9]]
        x, y = [0, 1, 0], 1
        dE, dU = gradients(E, U, x, y)
        eps = 1e-6
        for weights, dw in [(E,dE), (U,dU)]:
            for i in range(2):
                for d in range(2):
                    old = weights[i][d]
                    weights[i][d] = old + eps
                    plus = forward(E,U,x,y)[-1]
                    weights[i][d] = old - eps
                    minus = forward(E,U,x,y)[-1]
                    weights[i][d] = old
                    self.assertAlmostEqual(dw[i][d], (plus-minus)/(2*eps), places=7)

    def test_one_recorded_update(self):
        _, (a,b) = train(BASE, steps=(0,1))
        grad = gradients(a['E'], a['U'], *a['next_pair'])
        for key, rows in zip(('E','U'), grad):
            for old, delta, new in zip(a[key], rows, b[key]):
                self.assertEqual([v - RATE*g for v,g in zip(old,delta)], new)

    def test_counterfactual_starts_with_identical_weights(self):
        vocab = ' '.join(BASE + EDIT).split()
        for mode in ('cbow', 'skip'):
            _, a = train(BASE, mode=mode, vocab=vocab)
            _, b = train(EDIT, mode=mode, vocab=vocab)
            self.assertEqual(a[0]['E'], b[0]['E'])
            self.assertEqual(a[0]['U'], b[0]['U'])
            self.assertNotEqual(a[-1]['E'], b[-1]['E'])
            self.assertLess(a[-1]['loss'], a[0]['loss'])
            self.assertLess(b[-1]['loss'], b[0]['loss'])


if __name__ == '__main__':
    unittest.main()
