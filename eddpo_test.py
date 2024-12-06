from trainer.eddpo import EDDPOTrainer
if __name__ == "__main__":
    trainer = EDDPOTrainer("./outputs/edm_qm9")
    trainer.step(0,0)