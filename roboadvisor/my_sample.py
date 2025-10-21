### "rlog_size": 44*2,
""" pra = PyRoboAdvisor(
    p,
    1000,
    "2025-07-18",
    {"CNC": 3,
     "MLGO": 546,
     "SYTA": 75,
     "NVTS": 54,}
)  """

pra = PyRoboAdvisor(p)

additional_tickers_2 = [
    "ASML",
    "APLD",
    "DOCU",
    "HIMS",
    "CRWV",
    "PLUN",
    "CORZ",
    "QTUM",
    "ONDS",
    "LAES",
    "NVTS",
    "SOFI",
    "SYTA",
    "MLGO",
    "REKR",
    "CGTX",
    "ABVE",
    "MRSN",
    "MVST",
    "EVTV",
    "DRO",
    "GLNG",
    "SLDP",
    "EOSE",
    "NRBT",
    "NBIS",
    "HOTH",
    "OPEN",
]
extras = [ "GLNG", "DOCU" ]
additional_tickers = [
  "TSM", "GLOB", "NVTS", "WGS", "LOAR", "MYRG", "EXPD", "OLED",
  "SNOW", "AXON", "WTS", "RSG", "SAP", "ASML", "SOFI", "RMS.PA", "SAN.MC"]

em_tickets = ["BABA", "TCEHY", "HDB", "IBN", "MELI"]
pra.readTickersFromWikipedia(additional_tickers)
pra.completeTickersWithIB()  # Completa los tickers de IB que no están en el SP500, para que pueda venderlos

pra.prepare()  # Prepara los datos y la estrategia
pra.simulate()

pra.automatizeOrders()



    def readTickersFromWikipedia(self, additional_tickers=None):
        # Leer la tabla de Wikipedia
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tablas = read_html_like(url)
        sp500 = tablas[0]  # La primera tabla es la que contiene la información

        # Obtener la columna de los símbolos/tickers
        # Aportación de @Tomeu
        tickers = sp500['Symbol'].str.replace('.', '-').tolist()
        if additional_tickers:
            tickers.extend(additional_tickers)
            print(f"Se han añadido {len(additional_tickers)} tickers adicionales.")