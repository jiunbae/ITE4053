from tqdm import tqdm

from nnn import utils


def main(args) -> float:

    if args.mode == 'tf':
        import tensorflow.keras as nn
    else:
        import nnn as nn

    score = .0

    with tqdm(total=args.repeat) as t:
        for e in range(args.repeat):
            train = utils.Dataset(args.size)
            test = utils.Dataset(args.size)

            model = nn.Sequential([
                nn.layers.Dense(2, input_dim=2, activation=getattr(nn.activations, 'sigmoid')),
                nn.layers.Dense(1, input_dim=2, activation=getattr(nn.activations, 'sigmoid')),
            ])

            optimizer = getattr(nn.optimizers, args.optimizer)(lr=args.lr)

            model.compile(optimizer=optimizer,
                          loss=args.loss,
                          metrics=['accuracy'])

            model.fit(train.X, train.Y, epochs=args.epoch, verbose=False)
            loss, acc = model.evaluate(test.X, test.Y, verbose=False)
            score += acc

            t.set_postfix(loss=f'{loss:.4f}',
                          score=f'{acc*100:.2f}%',
                          mean=f'{score/(e+1)*100:.2f}%')
            t.update()

    return score / args.repeat


if __name__ == '__main__':
    main(utils.arguments)
