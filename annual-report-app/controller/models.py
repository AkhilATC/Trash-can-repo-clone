from factory import db


class Data(db.Model):

    id = db.Column(db.Integer, primary_key=True)
    oil = db.Column(db.Integer)
    gas = db.Column(db.Integer)
    brine = db.Column(db.Integer)

    def __repr__(self):
        return '<Data %r>' % self.id

