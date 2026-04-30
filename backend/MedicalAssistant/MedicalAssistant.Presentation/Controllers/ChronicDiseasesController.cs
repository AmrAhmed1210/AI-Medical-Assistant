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
    public class ChronicDiseasesController : ControllerBase
    {
        private readonly IPatientRecordService _patientRecordService;

        public ChronicDiseasesController(IPatientRecordService patientRecordService)
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

        // GET /api/patients/{id}/chronic-diseases  (Doctor, Patient(own))
        [HttpGet("patients/{id:int}/chronic-diseases")]
        [Authorize(Roles = "Doctor,Patient")]
        public async Task<IActionResult> GetForPatient(int id)
        {
            if (!IsDoctor() && !IsOwnPatient(id)) return Forbid();

            var items = await _patientRecordService.GetChronicDiseasesAsync(id);
            var dto = items.Select(x => new ChronicDiseaseMonitorDto(
                x.Id,
                x.PatientId,
                x.DiseaseName,
                x.DiseaseType,
                x.DiagnosedDate,
                x.Severity,
                x.IsActive,
                x.DoctorNotes,
                x.TargetValues,
                x.MonitoringFrequency,
                x.LastCheckDate,
                x.CreatedAt,
                x.UpdatedAt
            ));

            return Ok(dto);
        }

        // POST /api/patients/{id}/chronic-diseases  (Doctor)
        [HttpPost("patients/{id:int}/chronic-diseases")]
        [Authorize(Roles = "Doctor")]
        public async Task<IActionResult> Create(int id, [FromBody] CreateChronicDiseaseMonitorDto dto)
        {
            var entity = new ChronicDiseaseMonitor
            {
                DiseaseName = dto.DiseaseName,
                DiseaseType = dto.DiseaseType,
                DiagnosedDate = dto.DiagnosedDate,
                Severity = dto.Severity,
                MonitoringFrequency = dto.MonitoringFrequency,
                DoctorNotes = dto.DoctorNotes,
                TargetValues = dto.TargetValues,
                LastCheckDate = dto.LastCheckDate,
                IsActive = dto.IsActive ?? true,
            };

            var created = await _patientRecordService.AddChronicDiseaseAsync(id, entity);

            return CreatedAtAction(nameof(GetForPatient), new { id }, new ChronicDiseaseMonitorDto(
                created.Id,
                created.PatientId,
                created.DiseaseName,
                created.DiseaseType,
                created.DiagnosedDate,
                created.Severity,
                created.IsActive,
                created.DoctorNotes,
                created.TargetValues,
                created.MonitoringFrequency,
                created.LastCheckDate,
                created.CreatedAt,
                created.UpdatedAt
            ));
        }

        // PATCH /api/chronic-diseases/{id}  (Doctor)
        [HttpPatch("chronic-diseases/{id:int}")]
        [Authorize(Roles = "Doctor")]
        public async Task<IActionResult> Update(int id, [FromBody] UpdateChronicDiseaseMonitorDto dto)
        {
            var updates = new ChronicDiseaseMonitor
            {
                DiseaseName = dto.DiseaseName ?? string.Empty,
                DiseaseType = dto.DiseaseType ?? string.Empty,
                DiagnosedDate = dto.DiagnosedDate,
                Severity = dto.Severity ?? string.Empty,
                MonitoringFrequency = dto.MonitoringFrequency ?? string.Empty,
                DoctorNotes = dto.DoctorNotes,
                TargetValues = dto.TargetValues,
                LastCheckDate = dto.LastCheckDate,
                IsActive = dto.IsActive ?? true,
            };

            var updated = await _patientRecordService.UpdateChronicDiseaseAsync(id, updates);
            if (updated == null) return NotFound(new { message = "Condition not found." });

            return Ok(new ChronicDiseaseMonitorDto(
                updated.Id,
                updated.PatientId,
                updated.DiseaseName,
                updated.DiseaseType,
                updated.DiagnosedDate,
                updated.Severity,
                updated.IsActive,
                updated.DoctorNotes,
                updated.TargetValues,
                updated.MonitoringFrequency,
                updated.LastCheckDate,
                updated.CreatedAt,
                updated.UpdatedAt
            ));
        }

        // DELETE /api/chronic-diseases/{id}  (Doctor,Admin)
        [HttpDelete("chronic-diseases/{id:int}")]
        [Authorize(Roles = "Doctor,Admin")]
        public async Task<IActionResult> Deactivate(int id)
        {
            var ok = await _patientRecordService.DeactivateChronicDiseaseAsync(id);
            if (!ok) return NotFound(new { message = "Condition not found." });
            return Ok(new { message = "Condition deactivated." });
        }
    }
}
