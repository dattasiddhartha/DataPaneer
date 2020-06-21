import { HttpClient } from '@angular/common/http';
import { Component, ViewChild, ElementRef } from '@angular/core';
import { MatIconRegistry } from "@angular/material/icon";
import { map } from 'rxjs/operators';
import { listData } from './shared/list';


@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  list = listData.reverse();
  @ViewChild('searchbar') searchbar: ElementRef;
  searchText = '';

  toggleSearch: boolean = false;
  constructor() {

  }

  openSearch() {
    this.toggleSearch = true;
    this.searchbar.nativeElement.focus();
  }
  searchClose() {
    this.searchText = '';
    this.toggleSearch = false;
  }
}
