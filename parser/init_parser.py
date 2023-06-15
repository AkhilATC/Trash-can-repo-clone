import click
from art import text2art
import subprocess


@click.command()
def init_parser():
    title = text2art('EquipoAssign parser', font='big')
    click.echo(click.style(title, blink=True, fg='green'))
    click.echo(click.style("Parse HCPCS code from https://www.hcpcsdata.com/ \n"))
    prompt_msg = click.style("💬 : Please enter your output file name (default: output)", fg='blue')
    file_name = click.prompt(prompt_msg, type=str) or "output"
    click.echo("Select ouput format: \n")
    click.echo(click.style("[1] JSON \n", fg='red'))
    click.echo(click.style("[2] CSV \n", fg='red'))
    user_input = click.prompt(click.style('Select your option', fg='blue'), type=int)
    click.echo(click.style("Scraping initiated...... 🚀 ", fg='red'))
    if user_input == 1:
        subprocess.call([f"scrapy runspider parser.py -o {file_name}.json:json"], shell=True)
    elif user_input == 2:
        subprocess.call([f"scrapy runspider parser.py -o {file_name}.csv:csv"], shell=True)
    else:
        click.echo("You entered wrong input.....")
        click.echo(click.style('OK GoodBye', fg='yellow'))


if __name__ == '__main__':
    init_parser()