const assert = require("node:assert/strict");
const fs = require("node:fs");
const {chromium} = require("playwright");

async function main() {
  const browser = await chromium.launch({
    executablePath: process.argv[3],
    args: ["--no-sandbox", "--disable-breakpad", "--disable-crash-reporter", "--no-zygote"],
    timeout: 10000,
  });
  try {
    const page = await browser.newPage();
    const errors = [], network = [];
    page.on("pageerror", error => errors.push(error.message));
    page.on("request", request => network.push(request.url()));
    await page.setContent(fs.readFileSync(process.argv[2], "utf8"), {waitUntil: "domcontentloaded"});
    const variables = page.locator("#variableTable");
    await page.getByRole("button", {name: /Eligible variables/}).click();
    assert.equal(await page.locator("#state").inputValue(), "all");
    assert.match(await variables.getByRole("status").textContent(), /Page 1 of \d+/);
    await variables.getByRole("button", {name: "Next"}).click();
    assert.match(await variables.getByRole("status").textContent(), /Page 2 of/);
    await page.locator("#state").selectOption("pending");
    await page.getByRole("button", {name: "Reset filters"}).click();
    await page.locator("#search").fill("1300");
    assert.match(await variables.getByRole("status").textContent(), /1 rows/);
    await page.locator("#namespace").selectOption("ext");
    assert.equal(await variables.getByRole("status").textContent(), "No rows");
    await page.locator("#namespace").selectOption("all");
    await variables.getByRole("button", {name: "lab__anti_ro60__text", exact: true}).click();
    assert.equal(await page.locator("#which").inputValue(), "lab__anti_ro60__text");
    const detail = page.locator("#detail");
    await detail.getByLabel("Result version").selectOption("original");
    assert.equal(await detail.getByRole("columnheader", {name:"V2 typed result"}).count(),0);
    await detail.getByLabel("Result version").selectOption("v1");
    assert.equal(await detail.getByRole("columnheader", {name:"V1 structured"}).count(),1);
    await detail.getByLabel("Result version").selectOption("v2");
    assert.equal(await detail.getByRole("columnheader", {name:"V2 typed result"}).count(),1);
    await detail.getByRole("button", {name:"Evidence", exact:true}).click();
    assert.equal(await page.locator("#which").inputValue(), "lab__anti_ro60__text");
    await detail.getByRole("button", {name:"Before / V1 / V2"}).click();
    await page.locator("#which").selectOption("lab__other__text");
    await page.locator("#kind").selectOption("ana");
    assert.match(await page.locator("#specialTable [role=status]").textContent(), /Page 1 of/);

    await page.locator("#pendingSearch").fill("</script>");
    await page.getByRole("button",{name:"Expand page"}).click();
    await page.locator("#pendingTable mark").first().waitFor();
    assert.match(await page.locator("#pendingTable mark").first().textContent(), /<\/script>/);
    assert.equal(await page.locator("img").count(), 0);
    const exported = await page.evaluate(async () => {
      let blob;
      const oldCreate = URL.createObjectURL;
      const oldClick = HTMLAnchorElement.prototype.click;
      URL.createObjectURL = value => { blob = value; return "blob:report-test"; };
      HTMLAnchorElement.prototype.click = function () {};
      const click = label => [...document.querySelectorAll("button")].find(button => button.textContent === label).click();
      try {
        const payload = JSON.parse(document.getElementById("payload").textContent);
        click("All mappings JSON");
        const all = JSON.parse(await blob.text());
        click("Variable CSV");
        const csv = await blob.text();
        click("Filtered backlog JSON");
        const backlog = JSON.parse(await blob.text());
        click("Total backlog CSV");
        return {expected:payload.mappings.length,total:all.length,csv,backlog,full:await blob.text(),injection:csvCell(" \t=1+1")};
      } finally {
        URL.createObjectURL = oldCreate;
        HTMLAnchorElement.prototype.click = oldClick;
      }
    });
    assert.equal(exported.total,exported.expected);
    assert.match(exported.csv, /<\/script><img/);
    assert.match(exported.injection, /^"'/);
    assert.equal(exported.backlog.length,1);
    assert.match(exported.full,/derived_rule_or_availability_missing/);
    await page.setViewportSize({width:375,height:800});
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    assert.deepEqual(network, []);
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
