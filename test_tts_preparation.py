from carag.services.bhasini.tts_chunker import chunk_tts_text
from carag.services.bhasini.tts import prepare_tts_text
your_response = """
**Pest‑control guidelines extracted from the uploaded documents**

---

### 1.  Classification of approved insecticides  

| Category | Source pages (within the PDF) | Key points |
|----------|------------------------------|------------|
| **Agricultural use** | 02–58 (single compounds) and 59–66 (combinations) | Insecticides registered for crop protection; dosage and waiting‑period tables are provided. | [2] |
| **Public‑health use** | 67–72 | Insecticides for vector control (mosquitoes, flies, etc.) with specific application rates and safety intervals. | [2] |
| **Household use** | 73–84 | Ready‑to‑use formulations (coils, sprays, gels) for indoor pest control. | [2] |
| **Ad‑hoc approvals** | 85 | Special approvals for Fall Army Worm (FAW) with recommended dosages. | [2] |

---

### 2.  Agricultural‑use guidelines  

| Pest | Insecticide | Dose (ha) | Waiting period (days) | Notes |
|------|-------------|-----------|-----------------------|-------|
| Red spider mites (Tetranychus urticae) | **Abamectin 01.90 % EC** | 500 ha | 3 | Dilution 0.025–0.050 % in water. | [2] |
| Cotton jassids | **Acephate 75 % SP** | 500–1000 ha | 15 | | [2] |
| Cotton bollworms | **Acephate 97 % DF** | 500–1000 ha | 48 | | [2] |
| Cotton bollworms (SC) | **Alphacypermethrin 10 % SC** | 500–1000 ha | 10 | | [2] |
| Cotton bollworms (WP) | **Emamectin benzoate 01.90 % EC** | 500 ha | 15 | | [4] |
| Cotton bollworms (WP) | **Alphacypermethrin 10 % WP** | 190–220 kg/ha | 10 | | [4] |
| Cotton bollworms (SC) | **Alphacypermethrin 10 % SC** | 250–300 kg/ha | 10 | | [4] |
| Cotton bollworms (SC) | **Alphacypermethrin 10 % SC** | 250–300 kg/ha | 10 | | [4] |
| *Other crops* (e.g., okra, chilli, brinjal) | Various insecticides (e.g., **Azadirachtin**, **Fenazaquin**, **Fipronil**) | Dosage ranges 500–1000 kg/ha | Waiting periods 3–30 days | See tables in chunk 4. | [4] |

*General application note:*  
- For many WP formulations, dilute 250–400 g in 10 L water to cover 500 m².  
- For SC formulations, mix 20 mL in 1 L water and spray 25–50 mL m⁻² depending on surface porosity. | [1] |

---

### 3.  Public‑health (vector‑control) guidelines  

| Pest | Insecticide | Dose (L ha) | Waiting period (days) | Application |
|------|-------------|-------------|-----------------------|-------------|
| Adult mosquitoes (Anopheles, Aedes, Culex) | **Lambda‑cyhalothrin 10 % WP** | 500 m³ | 3 | Dilution in water; 10 L per 500 m². | [1] |
| Mosquito larvae | **Pirimiphos‑methyl 01 % EC** | 25 mL ha | 1 | Spot spray in breeding habitats. | [3] |
| Mosquito larvae | **Temephos 50 % EC** | 37.5–125 mL ha | 1 | Pond, swamp, drainage. | [1] |
| Adult mosquitoes (household) | **D‑Trans Allethrin 02 %** | 0.5 g m⁻² | 1 | Mosquito mat. | [3] |
| Adult mosquitoes (household) | **D‑Trans Allethrin 00.10 %** | 0.1 g m⁻² | 1 | Mosquito coil. | [3] |

*Waiting periods* are typically 1 day for household products, while agricultural formulations may require 3–30 days before crop harvest. | [1][3] |

---

### 4.  Household‑use guidelines  

| Product | Active ingredient | Formulation | Target pest | Typical dosage | Application method |
|---------|-------------------|-------------|------------|----------------|--------------------|
| **Alphacypermethrin 0.50 % Chalk** | 0.50 % | Chalk | Cockroaches | 25–50 mg m⁻² | Dust or spray |
| **D‑Trans Allethrin 02 % Mosquito Mat** | 02 % | Mat | Adult mosquitoes | 0.5 g m⁻² | Place in rooms |
| **Emamectin benzoate 0.10 % Gel** | 0.10 % | Gel | American cockroach | 0.001 g a.i./m² (1 g gel/m²) | Spot or cracks |
| **Fipronil 00.03 % Gel** | 0.03 % | Gel | German & American cockroaches | 0.03 g (in bait gun) | Spot spray |
| **Imidacloprid 02.15 % Gel** | 02.15 % | Gel | Cockroaches | 0.03 g a.i./m² | Spot or cracks |
| **Propoxur 20 % EC** | 20 % | EC | Mosquitoes, flies, cockroaches | 0.03 g a.i./m² | Spray |
| **D‑Trans Allethrin 00.10 % Mosquito Coil** | 0.10 % | Coil | Mosquitoes | 0.1 g m⁻² | Coil burning |

*General safety notes:*  
- All household products are “ready‑to‑use” and require no dilution.  
- For gel baits, apply 1 g per spot (2–4 spots m⁻²) in cracks or crevices. | [3][5] |

---

### 5.  Ad‑hoc approvals (Fall Army Worm)

| Insecticide | Dose (ha) | Notes |
|-------------|-----------|-------|
| **Chlorantraniliprole 9.3 % + Lambda‑cyhalothrin 4.6 %** | 35 g a.i./ha | Combination for FAW. | [3] |
| **Cyantraniliprole 19.8 % + Thiamethoxam 19.8 %** | 2.38 g a.i./kg seed | Seed treatment. | [3] |
| **Spinetoram 11.7 % SC** | 30 g a.i./ha | Foliar spray. | [3] |
| **Emamectin benzoate 5 % + Lufenuron 40 % WG** | 36 g a.i./ha | Combination. | [3] |
| **Thiodicarb 75 % WP** | 750 g a.i./ha | Foliar spray. | [3] |
| **Novaluron 5.25 % + Emamectin benzoate 0.9 %** | 78.75 g a.i./ha | Combination. | [3] |

---

### 6.  General application & safety guidelines  

1. **Dilution** – For WP formulations, dilute 250–400 g in 10 L water; for SC, mix 20 mL in 1 L water.  
2. **Waiting period** – Agricultural products typically require 3–30 days before harvest; household products usually have a 1‑day waiting period.  
3. **Coverage** – Use the specified volume per hectare or per square metre to achieve uniform coverage.  
4. **Safety** – Follow label instructions for personal protective equipment (PPE) and avoid exposure to non‑target organisms.  

These guidelines summarize the key dosage, application, and safety information for the approved insecticides across agricultural, public‑health, household, and ad‑hoc use categories as presented in the uploaded PDF.
""" 



prepared = prepare_tts_text(your_response)

chunks = chunk_tts_text(prepared,max_chars=800)

print("Original characters:", len(your_response))
print("Prepared characters:", len(prepared))
print("Number of TTS chunks:", len(chunks))

for index, chunk in enumerate(chunks, start=1):
    print()
    print(f"----- CHUNK {index} -----")
    print("Characters:", len(chunk))
    print(chunk)

print("Original characters:", len(your_response))

print("TTS characters:", len(prepared))

print()

print(prepared)