import argparse

from .trainer import TrainingConfig, predict_edges, train_model, transfer_train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Open-source BRIDGE-GRN implementation')
    subparsers = parser.add_subparsers(dest='command', required=True)

    def add_common_train_args(p):
        p.add_argument('--expression', required=True)
        p.add_argument('--tf-list', required=True)
        p.add_argument('--train-edges', required=True)
        p.add_argument('--val-edges', required=True)
        p.add_argument('--test-edges', required=True)
        p.add_argument('--output-dir', required=True)
        p.add_argument('--lr', type=float, default=3e-3)
        p.add_argument('--epochs', type=int, default=20)
        p.add_argument('--batch-size', type=int, default=256)
        p.add_argument('--seed', type=int, default=2025)
        p.add_argument('--hidden1', type=int, default=128)
        p.add_argument('--hidden2', type=int, default=64)
        p.add_argument('--hidden3', type=int, default=32)
        p.add_argument('--output-dim', type=int, default=16)
        p.add_argument('--num-head1', type=int, default=3)
        p.add_argument('--num-head2', type=int, default=3)
        p.add_argument('--alpha', type=float, default=0.2)
        p.add_argument('--decoder', type=str, default='dot', choices=['dot', 'cosine', 'bilinear'])
        p.add_argument('--reduction', type=str, default='concate', choices=['concate', 'mean'])
        p.add_argument('--lambda-ctr', type=float, default=0.5)
        p.add_argument('--loop', action='store_true')
        p.add_argument('--edge-drop', type=float, default=0.2)
        p.add_argument('--pre-epochs', type=int, default=0)
        p.add_argument('--share-tower', action='store_true')
        p.add_argument('--svd-dim', type=int, default=0)
        p.add_argument('--no-normalize-expression', action='store_true')
        p.add_argument('--device', type=str, default='cpu')

    train_parser = subparsers.add_parser('train', help='Train BRIDGE-GRN from scratch')
    add_common_train_args(train_parser)

    transfer_parser = subparsers.add_parser('transfer', help='Fine-tune BRIDGE-GRN from a source checkpoint')
    add_common_train_args(transfer_parser)
    transfer_parser.add_argument('--checkpoint', required=True)

    predict_parser = subparsers.add_parser('predict', help='Predict scores for query TF-target pairs')
    predict_parser.add_argument('--checkpoint', required=True)
    predict_parser.add_argument('--expression', required=True)
    predict_parser.add_argument('--tf-list', required=True)
    predict_parser.add_argument('--support-edges', required=True)
    predict_parser.add_argument('--query-edges', required=True)
    predict_parser.add_argument('--output', required=True)
    predict_parser.add_argument('--device', type=str, default='cpu')
    return parser


def _config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        expression=args.expression,
        tf_list=args.tf_list,
        train_edges=args.train_edges,
        val_edges=args.val_edges,
        test_edges=args.test_edges,
        output_dir=args.output_dir,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        hidden3=args.hidden3,
        output_dim=args.output_dim,
        num_head1=args.num_head1,
        num_head2=args.num_head2,
        alpha=args.alpha,
        decoder=args.decoder,
        reduction=args.reduction,
        lambda_ctr=args.lambda_ctr,
        loop=args.loop,
        edge_drop=args.edge_drop,
        pre_epochs=args.pre_epochs,
        share_tower=args.share_tower,
        svd_dim=args.svd_dim,
        normalize_expression=not args.no_normalize_expression,
        device=args.device,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == 'train':
        train_model(_config_from_args(args))
        return
    if args.command == 'transfer':
        transfer_train(_config_from_args(args), checkpoint_path=args.checkpoint)
        return
    if args.command == 'predict':
        predict_edges(
            checkpoint_path=args.checkpoint,
            expression_path=args.expression,
            tf_list_path=args.tf_list,
            support_edges_path=args.support_edges,
            query_edges_path=args.query_edges,
            output_path=args.output,
            device_name=args.device,
        )
        return
    raise ValueError(f'Unknown command: {args.command}')


if __name__ == '__main__':
    main()
