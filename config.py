import torch

class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # random seeds for reproducibility
    SEED = 42
    
    IMG_SIZE = 224
    LABELS = ['PNEUMONIA', 'NORMAL']
    NUM_WORKERS = 2
    
    # model configuration
    INPUT_DIM = 768  # ViT feature dimension
    LATENT_DIM = 10  # Autoencoder latent dimension
    
    # training hyperparameters
    BATCH_SIZE = 32
    AE_BATCH_SIZE = 128
    AE_EPOCHS = 50
    AE_LEARNING_RATE = 0.001
    
    # XGBoost parameters
    XGB_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc', 'error'],
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.05,
        'reg_lambda': 1.0,
        'random_state': 42,
        'tree_method': 'hist',
    }
    
    # Paths
    MODELS_DIR = 'models'
    RESULTS_DIR = 'results'
    CHECKPOINTS_DIR = 'checkpoints'
