import { BrowserRouter, Routes, Route } from "react-router-dom";
import DashboardLayout from "./layouts/DashboardLayout";
import { AdminDashboard } from "../components/dashboard/AdminDashboard";
import { AdminHospitals } from "../components/dashboard/AdminHospitals";
import { AdminUsers } from "../components/dashboard/AdminUsers";
import { AdminStatistics } from "../components/dashboard/AdminStatistics";
import  AddUser  from "../pages/AddUser";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<DashboardLayout />}>
          <Route index element={<AdminDashboard />} />
          <Route path="hospitals" element={<AdminHospitals />} />
          <Route path="users" element={<AdminUsers />} />
          <Route path="statistics" element={<AdminStatistics />} />
          <Route path="add-user" element={<AddUser />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;