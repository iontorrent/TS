# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django.test import TestCase
from django.test.client import Client
from django.contrib.auth.models import User
from django.core.urlresolvers import reverse
from iondb.rundb.login.forms import UserRegistrationForm

class LoginTestCase(TestCase):
    fixtures = ['iondb/rundb/tests/views/report/fixtures/globalconfig.json', 
                'iondb/rundb/tests/models/fixtures/groups.json', 
                'iondb/rundb/tests/models/fixtures/users.json']
    def setUp(self):
        self.ionadmin = User.objects.get(username='ionadmin')

    def testLogin(self):
        self.client.login(username='ionadmin', password='ionadmin')
        response = self.client.get(reverse('login'))
        self.assertEqual(response.status_code, 200)

class UserRegistrationFormTest(TestCase):

    def test_form(self):
        form = UserRegistrationForm({'username': 'abc', 'email': 'abc@example.com', 'password1': '12345', 'password2': '12345'})
        self.assertTrue(form.is_bound)
        self.assertEqual(form.errors, {})
        self.assertTrue(form.is_valid())
        self.assertHTMLEqual(form.errors.as_ul(), '')
        self.assertEqual(form.errors.as_text(), '')
        self.assertEqual(form.cleaned_data["username"], 'abc')
        self.assertEqual(form.cleaned_data["email"], 'abc@example.com')
        self.assertEqual(form.cleaned_data["password1"], '12345')
        self.assertEqual(form.cleaned_data["password2"], '12345')
        self.assertHTMLEqual(str(form['username']), '<input type="text" maxlength="30" name="username" value="abc" id="id_username" />')
        self.assertHTMLEqual(str(form['email']), '<input type="text" maxlength="75" name="email" value="abc@example.com" id="id_email" />')
        self.assertHTMLEqual(str(form['password1']), '<input type="password" name="password1" id="id_password1" />')
        self.assertHTMLEqual(str(form['password2']), '<input type="password" name="password2" id="id_password2" />')
        try:
            form['nonexistentfield']
            self.fail('Attempts to access non-existent fields should fail.')
        except KeyError:
            pass

        form_output = []

        for boundfield in form:
            form_output.append(str(boundfield))

        self.assertHTMLEqual('\n'.join(form_output), """<input type="text" maxlength="30" name="username" value="abc" id="id_username" />
<input type="text" maxlength="75" name="email" value="abc@example.com" id="id_email" />
<input type="password" name="password1" id="id_password1" />
<input type="password" name="password2" id="id_password2" />""")

        form_output = []

        for boundfield in form:
            form_output.append([boundfield.label, boundfield.data])

        self.assertEqual(form_output, [
            ['Username', 'abc'],
            ['E-mail', 'abc@example.com'],
            ['Password', '12345'],
            ['Password (again)', '12345']
        ])
        self.assertHTMLEqual(str(form), """<tr><th><label for="id_username">Username:</label></th><td><input type="text" maxlength="30" name="username" value="abc" id="id_username" /></td></tr>
<tr><th><label for="id_email">E-mail:</label></th><td><input type="text" maxlength="75" name="email" value="abc@example.com" id="id_email" /></td></tr>
<tr><th><label for="id_password1">Password:</label></th><td><input type="password" name="password1" id="id_password1" /></td></tr>
<tr><th><label for="id_password2">Password (again):</label></th><td><input type="password" name="password2" id="id_password2" /></td></tr>""")


    def test_form_username_is_valid(self):
        valid = {'username': 'abAB123.@+-_', 'email': 'abc@example.com', 'password1': '12345', 'password2': '12345'}
        form = UserRegistrationForm(valid)
        self.assertTrue(form.is_bound)
        self.assertEqual(form.errors, {})
        self.assertTrue(form.is_valid())
        
        invalid = dict(valid)
        invalid['username'] = ' '
        form = UserRegistrationForm(invalid)
        self.assertTrue(form.is_bound)
        self.assertEqual(form.errors['username'], ['This value may contain only letters, numbers and @/./+/-/_ characters.'])
        self.assertEqual(form.errors.keys(), ['username'], 'Only username should have an error')
        self.assertFalse(form.is_valid())

    def test_form_password_mismatch(self):
        mismatch = {'username': 'abAB123.@+-_', 'email': 'abc@example.com', 'password1': '12345', 'password2': '12346'}
        form = UserRegistrationForm(mismatch)
        self.assertTrue(form.is_bound)
        self.assertEqual(form.errors, {'__all__': [u"The two password fields didn't match."]})
        self.assertFalse(form.is_valid())


class UserRegistrationFormExistingUserTest(TestCase):
    fixtures = ['iondb/rundb/tests/views/report/fixtures/globalconfig.json', 
                'iondb/rundb/tests/models/fixtures/groups.json', 
                'iondb/rundb/tests/models/fixtures/users.json']
    def setUp(self):
        self.ionadmin = User.objects.get(username='ionadmin')
    
    def test_form_username_already_used(self):
        valid = {'username': 'ionadmin', 'email': 'abc@example.com', 'password1': '12345', 'password2': '12345'}
        form = UserRegistrationForm(valid)
        self.assertTrue(form.is_bound)
        self.assertFalse(form.is_valid())
        self.assertEqual(form.errors, {'username': [u'A user with that username already exists.']})
        self.assertEqual(form.errors.keys(), ['username'], 'Only username should have an error')
