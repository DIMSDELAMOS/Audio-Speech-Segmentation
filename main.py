import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

def extract_features(file_path, frame_length=0.025, hop_length=0.010):
    """
    Διαβάζει ένα αρχείο ήχου και εξάγει χαρακτηριστικά (MFCCs και Ενέργεια) ανά frame.
    frame_length: Μήκος του κάθε frame σε δευτερόλεπτα (25ms είναι το standard)
    hop_length: Το βήμα που κάνει το frame σε δευτερόλεπτα (10ms)
    """
    print(f"Επεξεργασία: {file_path}")
    
    y, sr = librosa.load(file_path, sr=16000)
    
    n_fft = int(frame_length * sr)
    hop_length_samples = int(hop_length * sr)
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length_samples)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length_samples)
    
    features = np.vstack((mfccs, rms))
    
    return features.T

def create_dataset(base_dir):
    """
    Διαβάζει όλα τα αρχεία από τους υποφακέλους speech και noise,
    εξάγει χαρακτηριστικά και επιστρέφει το πλήρες dataset.
    """
    X = [] # Εδώ θα μαζέψουμε τα χαρακτηριστικά
    y = [] # Εδώ θα μαζέψουμε τις ετικέτες (1 ή 0)
    
    # Οι κατηγορίες μας και οι ετικέτες τους (1 για ομιλία, 0 για θόρυβο)
    classes = {'speech': 1, 'noise': 0}
    
    for class_name, label in classes.items():
        # Βρίσκουμε τον φάκελο (π.χ. train/speech)
        class_dir = os.path.join(base_dir, class_name) 
        
        if not os.path.exists(class_dir):
            print(f"Προειδοποίηση: Δεν βρέθηκε ο φάκελος {class_dir}")
            continue
            
        print(f"\n--- Φόρτωση αρχείων από: {class_name} ---")
        
        # Ψάχνουμε όλους τους υποφακέλους για αρχεία ήχου
        for root, dirs, files in os.walk(class_dir):
            for file in files:
                if file.endswith('.wav') or file.endswith('.flac'):
                    file_path = os.path.join(root, file)
                    try:
                        features = extract_features(file_path)
                        X.append(features)
                        
                        # Φτιάχνουμε έναν πίνακα με ετικέτες, μία για κάθε frame
                        labels = np.full(features.shape[0], label)
                        y.append(labels)
                    except Exception as e:
                        print(f"Σφάλμα στο αρχείο {file_path}: {e}")
                        
    # Ενώνουμε όλους τους μικρούς πίνακες
    if len(X) > 0 and len(y) > 0:
        X_train = np.vstack(X)
        y_train = np.concatenate(y)
        return X_train, y_train
    else:
        return None, None

if __name__ == "__main__":
    train_dir = "train" 
    
    print("Ξεκινάει η δημιουργία του Dataset. (Θα πάρει 1-2 λεπτά όπως πριν)...")
    X_full, y_full = create_dataset(train_dir)
    
    if X_full is not None:
        print("\n=== ΤΟ DATASET ΦΟΡΤΩΘΗΚΕ! ===")
        
        # 1. Κρατάμε ένα υποσύνολο (π.χ. 5% των δεδομένων) για να μην κρασάρει ο υπολογιστής
        # Το stratify=y_full εξασφαλίζει ότι θα έχουμε ίδια αναλογία speech/noise με το αρχικό
        print("Υποδειγματοληψία δεδομένων (Subsampling)...")
        _, X_subset, _, y_subset = train_test_split(X_full, y_full, test_size=0.05, stratify=y_full, random_state=42)
        
        # 2. Χωρίζουμε το υποσύνολο σε Train (80%) και Validation (20%) για να δούμε πώς τα πάει
        X_train, X_val, y_train, y_val = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)
        
        print(f"Θα εκπαιδεύσουμε τα μοντέλα σε {X_train.shape[0]} frames.")
        
        # 3. Κανονικοποίηση (Standardization) - Κρίσιμο για K-NN και MLP
        print("Κανονικοποίηση χαρακτηριστικών...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # --- 4. ΕΚΠΑΙΔΕΥΣΗ K-NN ---
        print("\n=== Εκπαίδευση K-NN (K=5) ===")
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1) # n_jobs=-1 για να χρησιμοποιήσει όλους τους πυρήνες
        knn.fit(X_train_scaled, y_train)
        
        y_pred_knn = knn.predict(X_val_scaled)
        print("Αποτελέσματα K-NN:")
        print(f"Accuracy: {accuracy_score(y_val, y_pred_knn):.4f}")
        
        # --- 5. ΕΚΠΑΙΔΕΥΣΗ MLP ---
        # 2 κρυφά στρώματα όπως ζητάει η εκφώνηση, με 64 και 32 νευρώνες αντίστοιχα
        print("\n=== Εκπαίδευση MLP (64, 32) ===")
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42, early_stopping=True)
        mlp.fit(X_train_scaled, y_train)
        
        y_pred_mlp = mlp.predict(X_val_scaled)
        print("Αποτελέσματα MLP:")
        print(f"Accuracy: {accuracy_score(y_val, y_pred_mlp):.4f}")


# --- 6. ΤΕΛΙΚΗ ΔΟΚΙΜΗ ΣΤΟ TEST ΑΡΧΕΙΟ & ΕΞΑΓΩΓΗ CSV ---
        from scipy.signal import medfilt
        import pandas as pd
        
        print("\n=== Εκκίνηση Τελικής Δοκιμής (Test Phase) ===")
        
        # Ψάχνουμε το μικτό αρχείο test (ακόμα και αν είναι κρυμμένο σε υποφακέλους)
        test_dir = "test"
        test_file_path = None
        for root, dirs, files in os.walk(test_dir):
            for file in files:
                if file.endswith('.wav') or file.endswith('.flac'):
                    test_file_path = os.path.join(root, file)
                    break
        
        if test_file_path:
            print(f"Βρέθηκε το αρχείο test: {test_file_path}")
            
            # 6.1 Εξαγωγή χαρακτηριστικών από το αρχείο test
            print("Εξαγωγή χαρακτηριστικών...")
            X_test_unscaled = extract_features(test_file_path)
            
            # 6.2 Κανονικοποίηση (ΜΕ ΤΟΝ ΙΔΙΟ SCALER της εκπαίδευσης!)
            X_test_scaled = scaler.transform(X_test_unscaled)
            
            # 6.3 Πρόβλεψη με το καλύτερο μοντέλο (MLP)
            print("Παραγωγή αποφάσεων ανά frame...")
            y_pred_test = mlp.predict(X_test_scaled)
            
            # 6.4 Μετα-επεξεργασία: Φίλτρο διάμεσης τιμής (Median Filter)
            # Εδώ "καθαρίζουμε" τα στιγμιαία λάθη (π.χ. 1 frame θορύβου μέσα σε 10 δευτερόλεπτα ομιλίας)
            print("Εφαρμογή μετα-επεξεργασίας...")
            y_pred_smoothed = medfilt(y_pred_test, kernel_size=31) # Περιττός αριθμός (π.χ. 31 frames = ~300ms)
            
            # 6.5 Μετατροπή των frames σε χρονικά όρια (δευτερόλεπτα)
            print("Μετατροπή σε χρονικά segments...")
            hop_length = 0.010 # Το βήμα που έχουμε ορίσει στην extract_features
            segments = []
            current_class = y_pred_smoothed[0]
            start_frame = 0
            filename_only = os.path.basename(test_file_path)
            
            for i in range(1, len(y_pred_smoothed)):
                if y_pred_smoothed[i] != current_class:
                    start_time = start_frame * hop_length
                    end_time = i * hop_length
                    label = "foreground" if current_class == 1 else "background"
                    segments.append([filename_only, round(start_time, 2), round(end_time, 2), label])
                    
                    current_class = y_pred_smoothed[i]
                    start_frame = i
                    
            # Προσθήκη του τελευταίου κομματιού
            end_time = len(y_pred_smoothed) * hop_length
            label = "foreground" if current_class == 1 else "background"
            segments.append([filename_only, round(start_frame * hop_length, 2), round(end_time, 2), label])
            
            # 6.6 Εξαγωγή σε μορφή CSV
            df = pd.DataFrame(segments, columns=["Audiofile", "start", "end", "class"])
            df.to_csv("results.csv", index=False)
            print(f"\n=== ΤΕΛΟΣ! Το αρχείο 'results.csv' δημιουργήθηκε με {len(df)} γραμμές! ===")
            
        else:
            print("\nΣφάλμα: Δεν βρέθηκε κανένα αρχείο ήχου στον φάκελο 'test'!")