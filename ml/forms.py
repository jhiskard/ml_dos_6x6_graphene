from django import forms
ATOM_CHOICES = (
    ('B', 'B'),
    ('N', 'N'),
)


class AtomForm(forms.Form):
    atomList = forms.CharField()
    atomCategory = forms.ChoiceField(choices=ATOM_CHOICES)
