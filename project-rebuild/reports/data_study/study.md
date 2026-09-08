# Raw data study

Rows: 7011 · Columns: 8

## Nulls, empties and junk literals (after normalisation)

| Column | Nulls | Unique |
|---|---|---|
| Complaint ID | 0 | 7011 |
| Date Received | 0 | 288 |
| Banking Product | 0 | 17 |
| Issue ID | 0 | 130 |
| Complaint Description | 0 | 6984 |
| State | 27 | 55 |
| ZIP | 30 | 3442 |
| Bank Response | 0 | 6 |


## Duplicates

| Check | Count |
|---|---|
| Identical rows | 0 |
| Repeated Complaint ID | 0 |
| Repeated description | 27 | 

## Raw labels

| Banking Product | Rows | Median words |
|---|---|---|
| Checking or savings account | 1655 | 165 |
| Credit card or prepaid card | 1233 | 168 |
| Mortgage | 848 | 200 |
| Debt collection | 748 | 105 |
| Credit reporting, credit repair services, or other personal consumer reports | 559 | 97 |
| Credit reporting | 550 | 95 |
| Money transfer, virtual currency, or money service | 411 | 199 |
| Credit card | 359 | 157 |
| Bank account or service | 256 | 180 |
| Student loan | 201 | 155 |
| Consumer Loan | 83 | 143 |
| Vehicle loan or lease | 35 | 196 |
| Payday loan | 30 | 94 |
| Prepaid card | 17 | 113 |
| Payday loan, title loan, or personal loan | 14 | 140 |
| Money transfers | 11 | 101 |
| Other financial service | 1 | 180 |

## Issue ID against product

Does each Issue ID belong to a single product? If so the issue is a nested sub-label and is a strong feature or a second target.

| Check | Count |
|---|---|
| Issue IDs | 130 |
| Issue IDs tied to exactly one product | 115 |
| Issue IDs spanning two or more products | 15 |

## Complaint text

| Statistic | Words | Chars |
|---|---|---|
| min | 2 | 14 |
| 1% | 12 | 68 |
| 25% | 81 | 431 |
| 50% | 148 | 796 |
| 75% | 272 | 1474 |
| 99% | 1098 | 5965 |
| max | 4687 | 28705 |

| Check | Rows |
|---|---|
| Under 10 words | 41 |
| Over 512 words (BERT truncation) | 591 |
| Uppercase only | 74 |
| Non-ASCII characters | 0 |
| Contain redaction tokens (XXXX) | 6168 |
| Contain money masks ({$...}) | 2717 |

Average redaction tokens per complaint: 13.4

## Most common redaction forms

| Token | Occurrences |
|---|---|
| XXXX | 62478 |
| XX/XX/XXXX | 7413 |
| XX/XX/XX23 | 1799 |
| XX | 1043 |
| XXXXXXXX | 645 |
| XX00 | 530 |
| XXXX/XXXX/XXXX | 461 |
| XX0 | 447 |

## Shortest complaints

| Product | Text |
|---|---|
| Checking or savings account | see attachment |
| Checking or savings account | Please see attached |
| Credit reporting | wrong employer info |
| Credit reporting, credit repair services, or other personal consumer reports | missed mortgage payment accusations |
| Checking or savings account | This account was open fraudulently. |
| Checking or savings account | I dont know what happened. |
| Credit card or prepaid card | I closed that credit card |
| Checking or savings account | I got scammed on XXXX |
| Credit card or prepaid card | Card replacement requested never received |
| Checking or savings account | Identity Fraud and Stolen Accounts |
| Credit reporting | Name is wrong on Equifax |
| Credit reporting | Unauthorized inquiry and/or unknown company |

## Repeated descriptions

| Times | Text (first 120 chars) |
|---|---|
| 4 | I am filing this complaint because Experian has ignored my request to provide me with the documents that their company h |
| 3 | I am upset, sad, distress on what is happening to my credit report. Some of my accounts has a late payment remark. I did |
| 2 | this debt is being continually handed to different debt collectors because Bank of ABC will not settle this debt on term |
| 2 | I have received multiple mail letters stating that my bank account was closed and checks have bounce in account I do not |
| 2 | I 've sent multiple letters to this agency about this not being my account. After being advised by identitytheft.gov I ' |
