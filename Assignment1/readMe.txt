initially the data is read using pandas dataframe and the messages are converted to a list

Only whole words are considered and are separated using advanced regex.

Words with clitics are considered as a single word in the second task.

Only 10/11(starting with a 0) are considered as mobile numbers. Irregular/Unconventional spacing between numbers have been ignored.

Emails in proper format, are considered. This resulted in 1 false positive case, but still is accepted because the format is globally accepted.

All monetory quantities, not part of any other word are considered.

Conventional emoticons, which also made sense in the dataset are considered, and rest are ignored.

Clitics as discussed on the classroom are considered. This

For task 7 and 8, the words are case sensitive. Specifically for case 7, no punctuations are included, and if desired have to be given as input.

For the classifications, multiple approaches have been applied and listed as well.
