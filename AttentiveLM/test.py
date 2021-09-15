from main import main

# Reproduces results from paper
# params_merity = ["--embedding-size", '400', '--hidden-size', '400', '--n-layers', '3',
#                  '--batch-size', '8', '--epochs', '350', '--seed', '123',
#                  '--log-interval', '200', '--patience', '5', '--lr', '30.0',
#                  '--rnn-dropout', '0.25', '--tie-weights', '--input-dropout',
#                  '0.4', '--file-name', 'merity', '--no-positional-attention']
# main(params_merity)


# params_attentive_lm = ["--embedding-size", '400', '--hidden-size', '400', '--n-layers', '2',
#                        '--batch-size', '16', '--epochs', '350', '--seed', '123',
#                        '--log-interval', '200', '--patience', '6', '--lr', '30.0',
#                        '--rnn-dropout', '0.2', '--tie-weights', '--file-name', 'salton',
#                        '--attention', '--no-positional-attention']
# main(params_attentive_lm)


positional_lm = ["--embedding-size", '400', '--hidden-size', '400', '--n-layers', '2',
                 '--batch-size', '16', '--epochs', '350', '--seed', '123', '--rnn-dropout', '0.2',
                 '--log-interval', '200', '--lr', '30.0', '--tie-weights', '--file-name',
                 'positional', '--patience', '5']
main(positional_lm)
