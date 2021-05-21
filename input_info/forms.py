from django import forms

class InfoForm(forms.Form):
    GENDER= [
    ('Male', 'Male'),
    ('Female', 'Female'),
    ('Non-binair', 'Non-binair'),
    ]
    RACE= [
    ('White', 'White'),
    ('black', 'African-american'),
    ('indian', 'Indian'),
    ('middle eastern', 'Middle eastern'),
    ('latino hispanic', 'Latino '),
    ('muslim', 'Muslim')
    ]
    SEXORIEN= [
    ('Straight', 'Straight'),
    ('Gay/Lesbian', 'Gay'),
    ('Gay/Lesbian', 'Lesbian'),
    ('Bisexual', 'Bisexual'),
    ]
    GENID= [
    ('Cis-gender', 'Cis-gender'),
    ('Non-binair', 'Non-binair'),
    ('Transgender', 'Transgender'),
    ]
    
    age = forms.IntegerField(widget=forms.TextInput(attrs={"placeholder": "Your age"}))
    gender= forms.CharField(label='What is your gender?', widget=forms.Select(choices=GENDER))
    race= forms.CharField(label='What is your race?', widget=forms.Select(choices=RACE))
    sexorien= forms.CharField(label='What is your sexual orientation?', widget=forms.Select(choices=SEXORIEN))
    genid= forms.CharField(label='What is your gender identity?', widget=forms.Select(choices=GENID))
    
    
    
    
    
