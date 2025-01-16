import { CompanyTypes, createScraper } from 'israeli-bank-scrapers';
import dotenv from 'dotenv';
import fs from 'fs/promises';

dotenv.config();

(async function () {
  try {
    const options = {
      companyId: CompanyTypes.otsarHahayal,
      startDate: new Date('2025-01-01'),
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
    console.error(`Scraping failed for the following reason: ${e.message}`);
  }
})();