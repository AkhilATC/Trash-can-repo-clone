import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ChandlerViewComponent } from './chandler-view.component';

describe('ChandlerViewComponent', () => {
  let component: ChandlerViewComponent;
  let fixture: ComponentFixture<ChandlerViewComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ChandlerViewComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(ChandlerViewComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
