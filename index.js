import { CompanyTypes, createScraper } from 'israeli-bank-scrapers';
import dotenv from 'dotenv';

dotenv.config();

(async function() {
  try {
    const options = {
      companyId: CompanyTypes.otsarHahayal,
      startDate: new Date('2024-05-01'),
      combineInstallments: false,
      showBrowser: true,
    };

    const credentials = {
      username: process.env.BANK_USERNAME,
      password: process.env.BANK_PASSWORD,
    };

    const scraper = createScraper(options);
    const scrapeResult = await scraper.scrape(credentials);

    if (scrapeResult.success) {
      scrapeResult.accounts.forEach((account) => {
        console.log(`Found ${account.txns.length} transactions for account number ${account.accountNumber}`);
      });
    } else {
      throw new Error(scrapeResult.errorType);
    }
  } catch (e) {
    console.error(`Scraping failed for the following reason: ${e.message}`);
  }
})();