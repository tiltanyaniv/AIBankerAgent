import { CompanyTypes, createScraper } from 'israeli-bank-scrapers';
import dotenv from 'dotenv';
import fs from 'fs/promises';

dotenv.config();

(async function () {
  try {
    const options = {
      headless: 'new',  // Ensures true headless mode
      args: [
        '--disable-gpu',
        '--disable-dev-shm-usage',
        '--disable-software-rasterizer',
        '--no-sandbox',
        '--disable-setuid-sandbox'
      ],
      companyId: CompanyTypes.leumi,
      startDate: new Date('2025-01-01'),
      combineInstallments: false,
      showBrowser: false,
    };
    
    const credentials = {
      username: process.env.BANK_USERNAME,
      password: process.env.BANK_PASSWORD,
    };

    const scraper = createScraper(options);
    const scrapeResult = await scraper.scrape(credentials);

    if (scrapeResult.success) {
      const transactions = [];

      scrapeResult.accounts.forEach((account) => {
        console.log(
          `Found ${account.txns.length} transactions for account number ${account.accountNumber}`
        );

        // Add transactions to the array
        transactions.push({
          accountNumber: account.accountNumber,
          transactions: account.txns,
        });
      });

      // Save transactions to a file
      const filePath = './transactions.json';
      await fs.writeFile(filePath, JSON.stringify(transactions, null, 2));
      console.log(`Transactions saved to ${filePath}`);
    } else {
      throw new Error(scrapeResult.errorType);
    }
  } catch (e) {
    console.error("Scraping failed with error:", e);  // Print full error object
    if (e.stack) console.error("Stack trace:", e.stack);
  }
})();