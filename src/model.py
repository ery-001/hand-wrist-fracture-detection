import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# --- V3 FIX: IMPORT FOCAL LOSS ---
from tensorflow.keras.losses import BinaryFocalCrossentropy

import ssl
import os

# This bypasses the SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

def build_model(input_shape=(224, 224, 3), learning_rate=0.001):
    """
    Constructs the transfer learning model with a Custom Spatial Attention Block
    and Focal Loss to prevent Majority Class Collapse.
    """
    base_model = DenseNet121(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape
    )
    
    base_model.trainable = False
    
    # ---------------------------------------------------------
    # THE SEMINAR FLEX: SPATIAL ATTENTION BLOCK
    # ---------------------------------------------------------
    # The output of DenseNet is a 7x7 grid of features. 
    x = base_model.output
    
    # 1. We create a 1x1 Convolution that looks at the grid and generates a "heatmap" of importance (0 to 1)
    attention_weights = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='attention_spotlight')(x)
    
    # 2. We multiply the original features by these attention weights. 
    # Healthy areas get multiplied by ~0 (ignored). Fractures get multiplied by ~1 (amplified).
    x = tf.keras.layers.Multiply()([x, attention_weights])
    # ---------------------------------------------------------
    
    # Proceed to the standard classification head
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x) 
    predictions = Dense(1, activation='sigmoid')(x) 
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # --- REVERTING TO STABLE CROSS-ENTROPY ---
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        # We drop the Focal Loss and return to standard binary_crossentropy
        loss='binary_crossentropy',
        metrics=[
            'accuracy', 
            tf.keras.metrics.Precision(name='precision'), 
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model

# Quick test block to ensure everything compiles if run directly
if __name__ == "__main__":
    print("Testing model compilation...")
    model = build_model()
    model.summary()
    print("Model architecture successfully built!")