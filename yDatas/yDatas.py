import os
import yfinance as yf

def ydatas():
    bist30_symbols = [
        "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "EKGYO.IS",
        "EREGL.IS", "FROTO.IS", "GARAN.IS", "GUBRF.IS", "HEKTS.IS",
        "KCHOL.IS", "KOZAA.IS", "KOZAL.IS", "KRDMD.IS", "MGROS.IS",
        "PGSUS.IS", "PETKM.IS", "SAHOL.IS", "SASA.IS", "SOKM.IS",
        "SISE.IS", "TCELL.IS", "THYAO.IS", "TKFEN.IS", "TOASO.IS",
        "TUPRS.IS", "VAKBN.IS", "YKBNK.IS", "BRSAN.IS", "ALARK.IS"
    ]

    output_folder = "yDatas/Raw"
    os.makedirs(output_folder, exist_ok=True)

    downloaded = []
    failed = []

    start_date = "2005-07-11"
    end_date = "2025-04-10"

    for symbol in bist30_symbols:
        file_path = os.path.join(output_folder, f"{symbol.replace('.IS', '')}.csv")
        
        try:
            df = yf.download(symbol, start=start_date, end=end_date)
            if not df.empty:
                df.to_csv(file_path, index=True)
                downloaded.append(symbol)
            else:
                failed.append(symbol)
        except Exception:
            failed.append(symbol)

    # Özet
    print("İşlem özeti:")
    if downloaded:
        print(f"İndirilenler: {', '.join(downloaded)}")
    if failed:
        print(f"Hata oluşanlar: {', '.join(failed)}")