import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule } from '@angular/forms';
import {
  MatToolbarModule,
  MatTabsModule,
  MatButtonModule,
  MatInputModule,
  MatIconModule,
  MatFormFieldModule,
  MatListModule
} from '@angular/material';
import { FlexLayoutModule } from '@angular/flex-layout';
import { AppComponent } from './app.component';
import { FilterPipe} from './shared/filter.pipe';

@NgModule({
  imports: [BrowserModule, FormsModule, MatToolbarModule,
  
    MatButtonModule, MatIconModule,
    MatTabsModule, FlexLayoutModule,
  MatFormFieldModule,MatListModule, MatInputModule],
  declarations: [AppComponent, FilterPipe],
  bootstrap: [AppComponent]
})
export class AppModule { }