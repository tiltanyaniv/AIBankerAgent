import { CompanyTypes, createScraper } from 'israeli-bank-scrapers';
import fs from 'fs/promises';



(async function () {
  try {
    const options = {
      headless: 'new',  // Ensures true headless mode
      verbose: true,
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
    
    let BANK_USERNAME = "default_username";
    let BANK_PASSWORD = "default_password";
    
    const credentials = {
      username: BANK_USERNAME,
      password: BANK_PASSWORD,
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
      // Log the full scrapeResult for more details
      console.error("Scrape result error details:", JSON.stringify(scrapeResult, null, 2));
      throw new Error(`Scraping failed: ${JSON.stringify(scrapeResult)}`);
    }
  } catch (e) {
    console.error("Scraping failed with error:", e);
    console.error("Complete error object:", JSON.stringify(e, Object.getOwnPropertyNames(e)));
    if (e.response) {
      console.error("Error response:", e.response);
      try {
        console.error("Error response data:", JSON.stringify(e.response.data));
      } catch (err) {
        console.error("Error response data:", e.response.data);
      }
    }
    if (e.request) {
      console.error("Error request:", e.request);
    }
    if (e.stack) {
      console.error("Stack trace:", e.stack);
    }
  }
})();