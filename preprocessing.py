import pandas as pd
import re

df = pd.read_csv("temizlenmis.csv")
df_postal = pd.read_csv("tr_kktc_postal_codes.csv", encoding="utf-8", sep=";")

# Province correction
il_listesi = df_postal["il"].dropna().unique().tolist()
temiz_iller = []
for il in il_listesi:
    temiz_il = str(il).lower().strip()
    if temiz_il and temiz_il != 'nan':
        temiz_iller.append(temiz_il)

# District correction
ilce_listesi = df_postal["ilce"].dropna().unique().tolist()
temiz_ilceler = []
for ilce in ilce_listesi:
    temiz_ilce = str(ilce).lower().strip()
    if temiz_ilce and temiz_ilce != 'nan':
        temiz_ilceler.append(temiz_ilce)

def il_adlarini_duzelt(text):
    if pd.isna(text) or text == '':
        return text
    
    text_lower = str(text).lower()
    
    for il in temiz_iller:
        il_chars = list(il)
        pattern = '\\s*'.join(re.escape(char) for char in il_chars)
        full_pattern = r'\\b' + pattern + r'\\b'
        
        if re.search(full_pattern, text_lower):
            text_lower = re.sub(full_pattern, il, text_lower)
    
    return text_lower

def ilce_adlarini_duzelt(text):
    if pd.isna(text) or text == '':
        return text
    
    text_lower = str(text).lower()
    
    for ilce in temiz_ilceler:
        ilce_chars = list(ilce)
        pattern = '\\s*'.join(re.escape(char) for char in ilce_chars)
        full_pattern = r'\\b' + pattern + r'\\b'
        
        if re.search(full_pattern, text_lower):
            text_lower = re.sub(full_pattern, ilce, text_lower)
    
    return text_lower

# Main cleaning function

def clean_turkish_address(address):

    if pd.isna(address) or not isinstance(address, str):
        return ""
    
    text = address.lower().strip()
    
    # 1. no: ifadeleri (no:17/5, no: 23A vb.)
    text = re.sub(r'\bno\s*:\s*\d+[/\-\\\w]*', '', text)
    
    # 2. 1-2 basamaklı sayıları kaldır (3+ basamaklı sayıları koru)
    text = re.sub(r'\b\d{1,2}[a-zA-Z]?\b', '', text)  
    
    # 1-2 basamaklı sayı + / + 1-2 basamaklı sayı ifadelerini kaldır. Örnek: (15/3, 7/2)
    text = re.sub(r'\b\d{1,2}/\d{1,2}\b', '', text)  
    
    # 1-2 basamaklı sayı + - + harfleri kaldır. Örnek: (15-A, 23-C)
    text = re.sub(r'\b\d{1,2}-[a-zA-Z]\b', '', text) 
    
    # 3. Kat, daire, kapı numarası kelimeleri
    unwanted_words = [
        r'\bkat\b', r'\bdaire\b', r'\bno\b', r'\bnumara\b', r'\bkapı\b',
        r'\bkat\s*:\s*\d*', r'\bdaire\s*:\s*\d*'
    ]
    for word in unwanted_words:
        text = re.sub(word, '', text)
    
    # 4. Apartman temizliği - apartmanı/apartman öncesi kelime ile birlikte sil
    text = re.sub(r'\b\w+\s+apartman[ıi]?\b', '', text)
    text = re.sub(r'\bapartman[ıi]?\s+\w+\b', '', text)
    
    # 5. Blok temizliği - tek harf + blok/blogu/bloğu
    text = re.sub(r'\b[a-z]\s+blok[uğ]?[ui]?\b', '', text)
    text = re.sub(r'\bblok[uğ]?[ui]?\s+[a-z]\b', '', text)
    text = re.sub(r'\b[a-z]\s*-\s*blok[uğ]?[ui]?\b', '', text)
    
    # 6. Yön belirteçleri
    direction_words = [
        r'\bkarşısı\b', r'\bkarşıs[ıi]\b', r'\byanı\b', r'\byan[ıi]\b', 
        r'\barkası\b', r'\barkas[ıi]\b', r'\bönu\b', r'\bönü\b',
        r'\biçi\b', r'\biç[ıi]\b', r'\baltı\b', r'\balt[ıi]\b',
        r'\büstü\b', r'\büst[ıi]\b'
    ]
    for word in direction_words:
        text = re.sub(word, '', text)
    
    # 7. Noktalama işaretlerini kaldır
    text = re.sub(r'[^\w\sğüşıöçĞÜŞIÖÇ]', ' ', text)
    
    # 8. Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

# Apply province and district correction
df['structured_address_duzeltilmis'] = df['structured_address'].apply(il_adlarini_duzelt)
df['structured_address_duzeltilmis'] = df['structured_address_duzeltilmis'].apply(ilce_adlarini_duzelt)

# Apply v2 cleaning
df['cleaned_address_v2'] = df['structured_address_duzeltilmis'].apply(clean_turkish_address)

# Remove empty addresses
df_final = df[df['cleaned_address_v2'].str.strip() != ''].copy()

# Save final result
df_final.to_csv("temizlenmis3.csv", index=False)

print(f"Processing complete!")
print(f"Original: {len(df)} rows")
print(f"Final: {len(df_final)} rows")
print(f"Saved to: temizlenmis3.csv")