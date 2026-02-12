![British Airways Logo](/assets/ba-logo.png)


## Task One: Modeling lounge eligibility at Heathrow Terminal 3
<p style="color: red; font-size: 16px; align:center;">What I learned</p>
<ul><li>How using airline data and modeling helps British Airways forecast lounge demand and plan for future capacity planning</li></ul>

<p style="color: red; font-size: 16px;">What I did</p>
<ul>
<li>Review lounge eligibility criteria and explore how customer groupings can inform lounge demand assumptions</li>
<li>Create a reusable lookup table and written justification that British Airways can apply to future flying schedules</li>
</ul>

<p><a href="/case-study/BA_Task_1.pdf">Click here to view project case study</a>

<h2 align="center" >Lounge Eligibility Lookup Table</h2>
<p>I have processed the dataset to generate the eligibility percentages. Here is a lookup table based on the analysis of the <a href="/spreadsheets/British Airways Summer Schedule Dataset - Forage Data Science Task 1 (1).xlsx">British Airways Summer Schedule</a> file.

<h4>Grouping Logic:</h4>
<ul>
<li><b>Total Capacity</b> was calculated as the sum of First, Business and Economy class seats.</li>
<li><b>Percentges</b> represent the total eligible passengers for that divided by the total seat capacity for that group.</li>
</ul>

![Lookup Table](/assets/lookup_table.png)

<a href="/spreadsheets/answers/Lounge_Eligibility_Lookup_Table.xlsx">Check out the full lookup table</a>

<p><b>Note:</b> These percentages are derived from the totals in your provided sample data. For example, roughly 1.2% of all seats on Long Haul North American flights are occupied by passengers eligible for the Concorde Room.</p>



<h3 align="center" >JUSTIFICATION</h3>

![Justification](/assets/justification.png)

<a href="/spreadsheets/answers/Lounge Eligibility Lookup Table.xlsx">Check out the full justification</a>