import os
import pandas as pd
from datetime import datetime
import yDatas 
import Preparative
import RawUpdater

bist30_symbols = [
    "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "EKGYO.IS",
    "EREGL.IS", "FROTO.IS", "GARAN.IS", "GUBRF.IS", "HEKTS.IS",
    "KCHOL.IS", "KOZAA.IS", "KOZAL.IS", "KRDMD.IS", "MGROS.IS",
    "PGSUS.IS", "PETKM.IS", "SAHOL.IS", "SASA.IS", "SOKM.IS",
    "SISE.IS", "TCELL.IS", "THYAO.IS", "TKFEN.IS", "TOASO.IS",
    "TUPRS.IS", "VAKBN.IS", "YKBNK.IS", "BRSAN.IS", "ALARK.IS"
]

today = datetime.now().date()

def check_data_integrity():
    
    raw_folder = "yDatas/Raw"
    bist_folder = "yDatas/Bist"
    test_folder = "yDatas/Test"
    
    missing_raw_symbols = []
    for symbol in bist30_symbols:
        file_path = os.path.join(raw_folder, f"{symbol.replace('.IS', '')}.csv")
        if not os.path.exists(file_path):
            missing_raw_symbols.append(symbol)

    needs_update_symbols = []
    weekday = datetime.now().weekday()
    if weekday in [5,6]:
        print("✓ Hafta sonu — veriler güncel.")
        needs_update_symbols = None
    else:
        if os.path.exists(raw_folder):
            for symbol in bist30_symbols:
                file_path = os.path.join(raw_folder, f"{symbol.replace('.IS', '')}.csv")
                if os.path.exists(file_path):
                    try:
                        df_raw = pd.read_csv(file_path)
                        if not df_raw.empty:
                            last_date_str = str(df_raw.iloc[-1]["Price"])
                            if last_date_str != str(today):
                                needs_update_symbols.append(symbol)
                    except Exception:
                        needs_update_symbols.append(symbol)

    missing_bist_symbols = []
    for symbol in bist30_symbols:
        file_path = os.path.join(bist_folder, f"{symbol.replace('.IS', '')}.csv")
        if not os.path.exists(file_path):
            missing_bist_symbols.append(symbol)

    missing_test_symbols = []
    for symbol in bist30_symbols:
        file_path = os.path.join(test_folder, f"{symbol.replace('.IS', '')}.csv")
        if not os.path.exists(file_path):
            missing_test_symbols.append(symbol)

    return {
        'missing_raw': missing_raw_symbols,
        'needs_update': needs_update_symbols,
        'missing_bist': missing_bist_symbols,
        'missing_test': missing_test_symbols
    }

def main():
    print("=== VERİ BÜTÜNLÜĞÜ KONTROLÜ BAŞLIYOR ===\n")
    
    status = check_data_integrity()

    print("Kontrol Sonuçları:")
    print(f"- Raw'da eksik: {len(status['missing_raw'])} sembol")
    if status["needs_update"] == None:
        print(f"- Güncelleme gereken: 0 sembol")
    else:
        print(f"- Güncelleme gereken: {len(status['needs_update'])} sembol")
    print(f"- Bist'te eksik: {len(status['missing_bist'])} sembol")
    print(f"- Test'te eksik: {len(status['missing_test'])} sembol")
    print()
    
    if status['missing_raw']:
        print(f">>> {len(status['missing_raw'])} sembol için temel veriler indiriliyor...")
        yDatas.ydatas()
        print()
    
    if status['needs_update']:
        print(f">>> {len(status['needs_update'])} sembol güncelleniyor...")
        RawUpdater.updater()
        print("\n>>> Raw ve güncellemeler birleştiriliyor...")
        RawUpdater.merger()
        print("\n>>> Güncel Raw sonrası Bist ve Test hazırlanıyor...")
        Preparative.prebist()
        Preparative.pretest()
        print()

    
    if status['missing_bist']:
        print(f">>> Bist klasöründe {len(status['missing_bist'])} sembol eksik, hazırlanıyor...")
        Preparative.prebist()
        print()
    
    if status['missing_test']:
        print(f">>> Test klasöründe {len(status['missing_test'])} sembol eksik, hazırlanıyor...")
        Preparative.pretest()
        print()

    if not any(status.values()): 
        print("✓ Tüm veriler güncel ve eksiksiz.") 
    else: 
        print("✓ Tüm gerekli işlemler tamamlandı.")

if __name__ == "__main__":
    main()