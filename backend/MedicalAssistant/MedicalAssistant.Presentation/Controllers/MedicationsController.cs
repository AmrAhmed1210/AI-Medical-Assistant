using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.PatientRecords;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System.Linq;
using System.Threading.Tasks;

namespace MedicalAssistant.Presentation.Controllers
{
    [ApiController]
    [Route("api")]
    [Authorize]
    public class MedicationsController : ControllerBase
    {
        private readonly IPatientRecordService _patientRecordService;

        public MedicationsController(IPatientRecordService patientRecordService)
        {
            _patientRecordService = patientRecordService;
        }

        private int? GetPatientIdFromToken()
        {
            var pid = User.FindFirst("PatientId")?.Value;
            return int.TryParse(pid, out var id) ? id : null;
        }

        private bool IsDoctor() => User.IsInRole("Doctor");

        private bool IsOwnPatient(int patientId)
        {
            var tokenPid = GetPatientIdFromToken();
            return tokenPid.HasValue && tokenPid.Value == patientId;
        }

        // GET /api/patients/{id}/medications  (Doctor,Patient(own))
        [HttpGet("patients/{id:int}/medications")]
        [Authorize(Roles = "Doctor,Patient")]
        public async Task<IActionResult> GetForPatient(int id)
        {
            if (!IsDoctor() && !IsOwnPatient(id)) return Forbid();

            var items = await _patientRecordService.GetMedicationsAsync(id, activeOnly: true);
            var dto = items.Select(m => new MedicationTrackerDto(
                m.Id,
                m.PatientId,
                m.PrescribedByDoctorId,
                m.ChronicDiseaseMonitorId,
                m.MedicationName,
                m.GenericName,
                m.Dosage,
                m.Form,
                m.Frequency,
                m.TimesPerDay,
                m.DoseTimes,
                m.StartDate,
                m.EndDate,
                m.Instructions,
                m.PillsRemaining,
                m.RefillThreshold,
                m.IsChronic,
                m.IsActive,
                m.CreatedAt
            ));

            return Ok(dto);
        }

        // POST /api/patients/{id}/medications  (Doctor)
        [HttpPost("patients/{id:int}/medications")]
        [Authorize(Roles = "Doctor")]
        public async Task<IActionResult> Create(int id, [FromBody] CreateMedicationTrackerDto dto)
        {
            var doctorIdClaim = User.FindFirst("DoctorId")?.Value;
            int? doctorId = int.TryParse(doctorIdClaim, out var did) ? did : null;

            var entity = new MedicationTracker
            {
                ChronicDiseaseMonitorId = dto.ChronicDiseaseMonitorId,
                MedicationName = dto.MedicationName,
                GenericName = dto.GenericName,
                Dosage = dto.Dosage,
                Form = dto.Form,
                Frequency = dto.Frequency,
                TimesPerDay = dto.TimesPerDay,
                DoseTimes = dto.DoseTimes,
                StartDate = dto.StartDate,
                EndDate = dto.EndDate,
                Instructions = dto.Instructions,
                PillsRemaining = dto.PillsRemaining,
                RefillThreshold = dto.RefillThreshold,
                IsChronic = dto.IsChronic,
                IsActive = dto.IsActive ?? true,
                PrescribedByDoctorId = doctorId,
            };

            var created = await _patientRecordService.AddMedicationAsync(id, entity);

            return Ok(new MedicationTrackerDto(
                created.Id,
                created.PatientId,
                created.PrescribedByDoctorId,
                created.ChronicDiseaseMonitorId,
                created.MedicationName,
                created.GenericName,
                created.Dosage,
                created.Form,
                created.Frequency,
                created.TimesPerDay,
                created.DoseTimes,
                created.StartDate,
                created.EndDate,
                created.Instructions,
                created.PillsRemaining,
                created.RefillThreshold,
                created.IsChronic,
                created.IsActive,
                created.CreatedAt
            ));
        }

        // PATCH /api/medications/{id}  (Doctor)
        [HttpPatch("medications/{id:int}")]
        [Authorize(Roles = "Doctor")]
        public async Task<IActionResult> Update(int id, [FromBody] UpdateMedicationTrackerDto dto)
        {
            var updates = new MedicationTracker
            {
                ChronicDiseaseMonitorId = dto.ChronicDiseaseMonitorId,
                MedicationName = dto.MedicationName ?? string.Empty,
                GenericName = dto.GenericName,
                Dosage = dto.Dosage ?? string.Empty,
                Form = dto.Form ?? string.Empty,
                Frequency = dto.Frequency ?? string.Empty,
                TimesPerDay = dto.TimesPerDay ?? 0,
                DoseTimes = dto.DoseTimes ?? string.Empty,
                StartDate = dto.StartDate ?? default,
                EndDate = dto.EndDate,
                Instructions = dto.Instructions,
                PillsRemaining = dto.PillsRemaining,
                RefillThreshold = dto.RefillThreshold ?? 0,
                IsChronic = dto.IsChronic ?? false,
                IsActive = dto.IsActive ?? true,
            };

            var updated = await _patientRecordService.UpdateMedicationAsync(id, updates);
            if (updated == null) return NotFound(new { message = "Medication not found." });

            return Ok(new MedicationTrackerDto(
                updated.Id,
                updated.PatientId,
                updated.PrescribedByDoctorId,
                updated.ChronicDiseaseMonitorId,
                updated.MedicationName,
                updated.GenericName,
                updated.Dosage,
                updated.Form,
                updated.Frequency,
                updated.TimesPerDay,
                updated.DoseTimes,
                updated.StartDate,
                updated.EndDate,
                updated.Instructions,
                updated.PillsRemaining,
                updated.RefillThreshold,
                updated.IsChronic,
                updated.IsActive,
                updated.CreatedAt
            ));
        }

        // DELETE /api/medications/{id}  (Doctor,Admin)
        [HttpDelete("medications/{id:int}")]
        [Authorize(Roles = "Doctor,Admin")]
        public async Task<IActionResult> Deactivate(int id)
        {
            var ok = await _patientRecordService.DeactivateMedicationAsync(id);
            if (!ok) return NotFound(new { message = "Medication not found." });
            return Ok(new { message = "Medication deactivated." });
        }

        // GET /api/patients/{id}/medications/schedule  (Patient(own),Nurse)
        [HttpGet("patients/{id:int}/medications/schedule")]
        [Authorize(Roles = "Patient,Nurse")]
        public async Task<IActionResult> GetTodaySchedule(int id)
        {
            if (User.IsInRole("Patient") && !IsOwnPatient(id)) return Forbid();

            var items = await _patientRecordService.GetTodayScheduleAsync(id);
            return Ok(items);
        }

        // GET /api/patients/{id}/medications/low-stock  (Doctor,Nurse)
        [HttpGet("patients/{id:int}/medications/low-stock")]
        [Authorize(Roles = "Doctor,Nurse")]
        public async Task<IActionResult> GetLowStock(int id)
        {
            var items = await _patientRecordService.GetLowStockMedicationsAsync(id);
            var dto = items.Select(m => new MedicationTrackerDto(
                m.Id,
                m.PatientId,
                m.PrescribedByDoctorId,
                m.ChronicDiseaseMonitorId,
                m.MedicationName,
                m.GenericName,
                m.Dosage,
                m.Form,
                m.Frequency,
                m.TimesPerDay,
                m.DoseTimes,
                m.StartDate,
                m.EndDate,
                m.Instructions,
                m.PillsRemaining,
                m.RefillThreshold,
                m.IsChronic,
                m.IsActive,
                m.CreatedAt
            ));

            return Ok(dto);
        }
    }
}
