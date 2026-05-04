import time
from scraper import JobScraper
from pdf_engine import generate_full_application

def main():
    print("🤖 Jules Master Bot gestartet...")

    # Initiiere den Scraper (headless=True für den Hintergrundbetrieb)
    scraper = JobScraper(headless=True)

    suchbegriff = "Werkstudent Maschinenbau"
    ort = "Köln" # Region Köln/Düsseldorf/Aachen

    alle_gefundenen_jobs = []

    # 1. Scraping von Jobbörsen
    print("\n🌐 Durchsuche Jobbörsen...")
    indeed_jobs = scraper.scrape_indeed(query=suchbegriff, location=ort)
    alle_gefundenen_jobs.extend(indeed_jobs)

    stepstone_jobs = scraper.scrape_stepstone(query=suchbegriff, location=ort)
    alle_gefundenen_jobs.extend(stepstone_jobs)

    # 2. Scraping direkter Unternehmensseiten (Wunschfirmen in der Region)
    print("\n🏭 Durchsuche Wunschfirmen...")
    wunschfirmen = ["Ford", "Bayer", "Henkel", "FEV", "Rheinmetall"]

    for firma in wunschfirmen:
        firmen_jobs = scraper.scrape_company_career_page(company=firma, query=suchbegriff)
        alle_gefundenen_jobs.extend(firmen_jobs)
        time.sleep(2) # Respektvolle Pause zwischen den Anfragen

    # 3. Bewerbungs-Pipeline starten
    print(f"\n📊 Insgesamt {len(alle_gefundenen_jobs)} Jobs gefunden.")
    print("🚀 Starte Dokumentengenerierung...")

    if not alle_gefundenen_jobs:
        print("Es wurden keine Jobs gefunden. Beende das Skript.")
        return

    # Wir durchlaufen maximal 5 Jobs als Demo
    for job in alle_gefundenen_jobs[:5]:
        generate_full_application(
            company=job['company'],
            job_title=job['title'],
            url=job['url']
        )
        time.sleep(1)

if __name__ == "__main__":
    main()
