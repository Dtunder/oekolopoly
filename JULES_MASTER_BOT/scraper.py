import time
from typing import List, Dict
from playwright.sync_api import sync_playwright

class JobScraper:
    """
    Ein leistungsstarker Scraper zum Extrahieren von dynamisch geladenen Job-Links
    von verschiedenen Job-Plattformen und Karriere-Seiten.
    """
    def __init__(self, headless: bool = True):
        self.headless = headless

    def scrape_indeed(self, query: str, location: str) -> List[Dict[str, str]]:
        """Scrapt Job-Angebote von Indeed."""
        print(f"[Indeed] Suche nach '{query}' in '{location}'...")
        jobs = []
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
            )
            page = context.new_page()

            url = f"https://de.indeed.com/jobs?q={query.replace(' ', '+')}&l={location.replace(' ', '+')}"
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)

                # Auf die Jobkarten warten
                page.wait_for_selector('a.jcs-JobTitle', timeout=15000)
                job_cards = page.locator('a.jcs-JobTitle').all()

                for card in job_cards:
                    title = card.text_content()
                    href = card.get_attribute("href")
                    if href:
                        full_link = f"https://de.indeed.com{href}" if href.startswith('/') else href
                        jobs.append({"title": title.strip(), "url": full_link, "company": "Indeed-Listing"})
            except Exception as e:
                print(f"[Indeed] Fehler beim Scrapen: {e}")
            finally:
                browser.close()

        return jobs

    def scrape_stepstone(self, query: str, location: str) -> List[Dict[str, str]]:
        """Scrapt Job-Angebote von StepStone."""
        print(f"[StepStone] Suche nach '{query}' in '{location}'...")
        jobs = []
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
            )
            page = context.new_page()

            url = f"https://www.stepstone.de/work/{query.replace(' ', '-')}/in-{location.replace(' ', '-')}"
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)

                # Cookie-Banner akzeptieren (falls vorhanden)
                try:
                    page.locator('button#ccmgt_explicit_accept').click(timeout=5000)
                except:
                    pass

                # Auf Jobkarten warten
                page.wait_for_selector('a[data-at="job-item-title"]', timeout=15000)
                job_cards = page.locator('a[data-at="job-item-title"]').all()

                for card in job_cards:
                    title = card.text_content()
                    href = card.get_attribute("href")
                    if href:
                        full_link = f"https://www.stepstone.de{href}" if href.startswith('/') else href
                        jobs.append({"title": title.strip(), "url": full_link, "company": "StepStone-Listing"})
            except Exception as e:
                print(f"[StepStone] Fehler beim Scrapen: {e}")
            finally:
                browser.close()

        return jobs

    def scrape_company_career_page(self, company: str, query: str) -> List[Dict[str, str]]:
        """
        Generische Suche nach Jobs auf Unternehmensseiten (z. B. Ford, Bayer, Henkel, FEV, Rheinmetall).
        Hier wird eine Google Dorking-Suche (site:company.com/careers) als Platzhalter genutzt,
        da jedes Unternehmen ein eigenes ATS (Workday, SAP SuccessFactors etc.) hat.
        """
        print(f"[{company}] Suche auf der Karriereseite nach '{query}'...")
        jobs = []
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
            )
            page = context.new_page()
            try:
                # Simuliert eine strukturierte Google-Suche für Karriere-Seiten
                search_query = f"site:{company.lower()}.com careers {query}"
                page.goto(f"https://www.google.com/search?q={search_query.replace(' ', '+')}", timeout=30000)

                # Google Cookie-Banner umgehen
                try:
                    page.locator('button:has-text("Alle akzeptieren"), button:has-text("Accept all")').click(timeout=3000)
                except:
                    pass

                page.wait_for_selector('div.g a', timeout=10000)
                results = page.locator('div.g a').all()

                for res in results[:3]: # Wir extrahieren die Top-3-Ergebnisse
                    title_loc = res.locator('h3')
                    title = title_loc.text_content() if title_loc.count() > 0 else "Unbekannter Job"
                    href = res.get_attribute('href')

                    if href and href.startswith('http'):
                        jobs.append({"title": title.strip(), "url": href, "company": company})

            except Exception as e:
                print(f"[{company}] Fehler bei der Suche: {e}")
            finally:
                browser.close()

        return jobs
