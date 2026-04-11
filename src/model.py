import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_model(input_shape=(224, 224, 3), learning_rate=0.001):
    """
    Constructs the transfer learning model using a pre-trained DenseNet121 backbone,
    appended with a custom classification head for binary fracture detection.
    """
    # 1. Load the pre-trained DenseNet121 backbone
    # include_top=False removes the original 1000-class ImageNet classification layer
    base_model = DenseNet121(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape
    )
    
    # Freeze the base model to retain the pre-learned ImageNet features
    base_model.trainable = False
    
    # 2. Build the Custom Classification Head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x) # 50% dropout to prevent overfitting
    predictions = Dense(1, activation='sigmoid')(x) # Binary output (0 or 1)
    
    # 3. Assemble the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # 4. Compile the model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
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