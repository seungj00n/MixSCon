"""Train the patient-aware CE or focal MixSCon variant."""
from mixscon.train_cli import run

def main():
    run('patient')

if __name__ == '__main__':
    main()
