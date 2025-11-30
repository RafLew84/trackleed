# Plan: YOLO + Hungarian z gatingiem biegunowym (wykonalny krok po kroku)

1. [x] Wyznaczanie pattern_center (zbiorczo z YOLO)
   - [x] Dla każdej klatki (YOLO detekcje ≥ 4) policz `find_pattern_center` i zbierz (cx, cy, err).
   - [x] Odrzuć outliery (duży błąd lub odległość od mediany).
   - [x] Ustal globalny center = mediana (cx, cy) z pozostałych i zapisz niepewność (np. MAD/σ); fallback: mediana pozycji detekcji jeśli brak wiarygodnych centrów.
2. [x] Nowy wariant trackingu „YOLO + Hungarian (polar)” w GUI
   - [x] Dodaj wybór algorytmu w combo (Tracking) + sekcję z parametrami: max Δr/klatkę, max Δθ (lub max ruch styczny), kierunek (do środka/na zewnątrz/oba), tolerancja brak dopasowania (N klatek).
   - [x] Pola wyboru źródła center: automatyczny (pkt 1) lub istniejący `pattern_center`.
3. [x] Implementacja nowego matchingu (osobna funkcja)
   - [x] Wejście: lista tracków (inicjalizowanych z wybranych YOLO ROI), detekcje YOLO na klatce t+1, center, parametry gatingu.
   - [x] Konwersja do (r, θ); odrzucenie par przekraczających progi (Δθ, Δr, zły znak Δr).
   - [x] Koszt: α·|Δr| + β·|Δθ|·r (β >> α); uruchom Hungarian na odfiltrowanej macierzy.
4. [ ] Śledzenie tylko wybranych plamek
   - [ ] Tracki startowe = ręcznie wybrane detekcje YOLO (brak tworzenia nowych tracków z nieprzypisanych detekcji).
   - [ ] Brak dopasowania przez >N klatek → track zamknięty (bez fallbacku OF/LK na razie).
5. [ ] Integracja i testy
   - [ ] Spiąć nowy wariant w `run_tracking`: ustaw center (pkt 1) i wywołaj nowy matcher.
   - [ ] Test na długiej sekwencji z pierścieniami: brak przeskoków między rzędami, stabilne ścieżki; ewentualna korekta progów domyślnych.
