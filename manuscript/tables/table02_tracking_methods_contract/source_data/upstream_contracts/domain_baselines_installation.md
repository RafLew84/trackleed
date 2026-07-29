# Instalacja domenowych baseline'ów LEED

Data weryfikacji: **2026-07-27**

Status: **EasyLEED i ViPErLEED zainstalowane; testy dymne oraz wspólny pilot
wejścia–wyjścia na danych `ScanE` zakończone powodzeniem.**

Pliki wykonywalne, środowiska i źródła znajdują się w ignorowanym przez Git
katalogu `third_party/` lub w lokalnym katalogu środowisk Conda. Nie są częścią
repozytorium ani przyszłego pakietu danych publikacji.

## EasyLEED

- źródło: <https://github.com/andim/easyleed>;
- commit źródła:
  `bbe4acd2b5eeb0f5d50a8d018780ace3afd7ff73`;
- wersja: `2.5.2`;
- licencja deklarowana przez projekt: `GPL-2.0`;
- źródło lokalne:
  `third_party/easyleed`;
- instalacja: editable z `third_party/easyleed/source`;
- środowisko:
  `C:\Users\rlewa\anaconda3\envs\easyleed`;
- plik konfiguracyjny utworzony przez aplikację:
  `C:\Users\rlewa\easyleed.ini`.

Istotne wersje środowiska:

| Składnik | Wersja |
|---|---:|
| Python | 3.11.15 |
| NumPy | 1.26.4 |
| SciPy | 1.17.1 |
| Matplotlib | 3.11.1 |
| pandas | 3.0.5 |
| scikit-image | 0.26.0 |
| Qt | 5.15.2 |
| PyQt | 5.15.11 |

NumPy został jawnie przypięty do `1.26.4`. Przy pierwotnie rozwiązanej wersji
`2.4.6` filtr Kalmana EasyLEED kończył się błędem konwersji jednowymiarowej
tablicy na `float`. Po przypięciu zależności:

- `pip check` zwraca `No broken requirements found`;
- import modułów i utworzenie `MainWindow` w trybie offscreen przechodzą;
- fit dwuwymiarowej funkcji Gaussa przechodzi;
- sześcioklatkowy test syntetycznego ruchu przechodzi z maksymalnym błędem
  pozycji `0.000000 px` dla bezszumowego przykładu kontrolnego.

Uruchomienie z Windows:

```text
C:\Users\rlewa\anaconda3\envs\easyleed\Scripts\easyleed.exe
```

Uruchomienie z WSL:

```bash
/mnt/c/Users/rlewa/anaconda3/envs/easyleed/Scripts/easyleed.exe
```

## ViPErLEED

- źródło: <https://github.com/viperleed/viperleed-imagej>;
- commit sklonowanego źródła:
  `2ed1044b72bb43a936f9456cdb8b9846e0d23e96`;
- wersja pluginu: `1.03`;
- licencja deklarowana przez projekt: `GPL-3.0-or-later`;
- przenośny runtime:
  `third_party/viperleed-runtime/ImageJ`;
- ImageJ: `1.54u`;
- Java: OpenJDK `1.8.0_345` z dystrybucji ImageJ.

Do przenośnej dystrybucji ImageJ dodano oficjalny release JAR ViPErLEED.
Dystrybucyjny `ij.jar` zachowano jako `ij.jar.distribution-2023`, a aktywny
`ij.jar` zastąpiono oficjalnym daily build, ponieważ ViPErLEED wymaga ImageJ
co najmniej `1.54k`.

Sumy SHA-256:

| Artefakt | SHA-256 |
|---|---|
| `ij154-win-java8.zip` | `32b05356d3a7906e10ca91491c2be06bfe0397c9e7a3d3b52fdad9e3ef48276c` |
| aktywny `ImageJ/ij.jar` | `6acc016e4630d5c5ecd470b205f8d1abc08ce57568c6ab617a308d2451d9c90d` |
| ViPErLEED plugin JAR | `dd058af9ea79eaf6bea837efc6188b8f9b67dadf980b540d84826a33062c8a2d` |

Test batchowy ImageJ:

1. potwierdził wersję ImageJ `1.54u`;
2. wykonał wymaganie `requires("1.54k")`;
3. odnalazł i uruchomił komendę pluginu `LEED Commands`.

Wynik: `ViPErLEED plugin smoke OK`.

Uruchomienie z Windows:

```text
C:\Users\rlewa\Documents\PROJEKTY\trackleed\third_party\viperleed-runtime\ImageJ\ImageJ.exe
```

Uruchomienie z WSL:

```bash
third_party/viperleed-runtime/ImageJ/ImageJ.exe
```

Po uruchomieniu tracker jest dostępny jako
`Plugins > ViPErLEED > LEED Spot Tracker...`.

## Pilot na rzeczywistych danych

Mały pilot wejścia–wyjścia na siedmiu klatkach `26a24_ScanE` został
wykonany dla tych samych trzech plam. Oba programy wyeksportowały natywne
pozycje, a propozycje zostały poprawnie przekazane do wspólnego ROI `48×48`
i fitu Gaussa.

Wyniki, adapter nazw energii, makro ViPErLEED i ograniczenia empirycznego
trzyplamowego wzoru opisuje
[raport pilota](domain-baseline-pilot/README.md).
