from brick_classifier import BrickClassifier

class ClassifierManager:
    def __init__(self, classifier_model_path, snapshots_folder):
        self.classifier = None
        self.classifier_model_path = classifier_model_path
        self.snapshots_folder = snapshots_folder


    def initialize_classifier(self):
        try:
            print("Initializing brick classifier...")
            self.classifier = BrickClassifier(self.classifier_model_path, self.snapshots_folder)
            print("Brick classifier initialized successfully!")
            return True
        except Exception as e:
            print(f"Failed to initialize classifier: {e}")
            return False


    def start_classifier(self, same_prediction_interval=5.0, check_interval=1.0):
        try:
            print(f"Starting continuous capture with {check_interval}s intervals...")
            self.classifier.start_continuous_capture(same_prediction_interval=same_prediction_interval, check_interval=check_interval)
            print("Classifier capture started!")
            return True
        except Exception as e:
            print(f"Failed to start classifier: {e}")
            return False


    def parse_classification(self, classification):
        """
        Parse classification string format: {color}_{shape}_{state}
        Returns: (brick_type, brick_size, color)
        where brick_type is the state ('damaged' or 'undamaged')
        """
        if not classification:
            return None, None, None
        
        try:
            parts = classification.split('_')
            if len(parts) != 3:
                print(f"Warning: Unexpected classification format: {classification}")
                return None, None, None
            
            color, shape, brick_type = parts
            
            # Normalize brick_type
            if brick_type.lower() in ['good']:
                brick_type = 'undamaged'
            else:
                brick_type = brick_type.lower()  # 'damaged', etc.
            
            # Use shape as brick_size
            brick_size = shape.lower()
            
            return brick_type, brick_size, color
        
        except Exception as e:
            print(f"Error parsing classification '{classification}': {e}")
            return None, None, None


    def get_latest_predictions(self):
        try:
            classes, confidences = self.classifier.get_latest_top4()
            return classes, confidences
        except Exception as e:
            print(f"Error getting classifier predictions: {e}")
            return None, None


    def cleanup(self):
        if self.classifier:
            try:
                self.classifier.cleanup()
                print("Classifier cleaned up")
            except Exception as e:
                print(f"Error cleaning up classifier: {e}")