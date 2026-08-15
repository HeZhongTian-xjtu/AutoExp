from train import main as train_main
from evaluator.evaluate import main as evaluate_main


if __name__ == '__main__':
    train_main(smoke=True)
    evaluate_main()
