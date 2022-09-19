import argparse


def parameter_parser():

    parser = argparse.ArgumentParser(description="Run MAGCNSE.")

    parser.add_argument("--dataset-path",
                        nargs="?",
                        default="../datasets",
                        help="Datasets.")

    parser.add_argument("--epoch",
                        type=int,
                        default=250,
                        help="Number of training epochs for representation learning. Default is 250.")

    parser.add_argument("--gcn-layers",
                        type=int,
                        default=2,
                        help="Number of GCN Layers. Default is 2.")

    parser.add_argument("--out-channels",
                        type=int,
                        default=128,
                        help="out-channels of CNN. Default is 128.")

    parser.add_argument("--lncRNA-number",
                        type=int,
                        default=489,
                        help="lncRNA number. Default is 489.")

    parser.add_argument("--fl",
                        type=int,
                        default=128,
                        help="lncRNA feature dimensions. Default is 128.")

    parser.add_argument("--disease-number",
                        type=int,
                        default=251,
                        help="disease number. Default is 251.")

    parser.add_argument("--fd",
                        type=int,
                        default=128,
                        help="disease feature dimensions. Default is 128.")

    parser.add_argument("--lncRNA-view",
                        type=int,
                        default=3,
                        help="lncRNA views number. Default is 3(3 datasets for lncRNA sim)")

    parser.add_argument("--disease-view",
                        type=int,
                        default=3,
                        help="disease views number. Default is 3(3 datasets for disease sim)")

    return parser.parse_args()
