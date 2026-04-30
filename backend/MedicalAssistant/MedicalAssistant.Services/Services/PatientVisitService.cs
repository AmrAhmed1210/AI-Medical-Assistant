using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.DoctorsModule;
using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Domain.Entities.UserModule;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.PatientVisits;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MedicalAssistant.Services.Services
{
    public class PatientVisitService : IPatientVisitService
    {
        private readonly IUnitOfWork _unitOfWork;

        public PatientVisitService(IUnitOfWork unitOfWork)
        {
            _unitOfWork = unitOfWork;
        }

        private async Task<Doctor?> GetDoctorByUserIdAsync(int doctorUserId)
        {
            var docs = await _unitOfWork.Repository<Doctor>().FindAsync(d => d.UserId == doctorUserId);
            return docs.FirstOrDefault();
        }

        private static PatientVisitDto MapVisit(PatientVisit v) => new(
            v.Id,
            v.PatientId,
            v.DoctorId,
            v.AppointmentId,
            v.ChiefComplaint,
            v.PresentIllnessHistory,
            v.ExaminationFindings,
            v.Assessment,
            v.Plan,
            v.Notes,
            v.Status,
            v.VisitDate,
            v.CreatedAt,
            v.ClosedAt
        );

        private static VisitSymptomDto MapSymptom(Symptom s) => new(
            s.Id,
            s.PatientVisitId,
            s.Name,
            s.Severity,
            s.Duration,
            s.Onset,
            s.Progression,
            s.Location,
            s.IsChronic,
            s.Notes,
            s.CreatedAt
        );

        private static VisitVitalDto MapVital(VisitVitalSign v) => new(
            v.Id,
            v.PatientId,
            v.PatientVisitId,
            v.Type,
            v.Value,
            v.Value2,
            v.Unit,
            v.IsAbnormal,
            v.NormalRangeMin,
            v.NormalRangeMax,
            v.RecordedBy,
            v.Notes,
            v.RecordedAt
        );

        public async Task<PatientVisitDto> OpenVisitAsync(int doctorUserId, CreateVisitDto dto)
        {
            var doctor = await GetDoctorByUserIdAsync(doctorUserId) ?? throw new UnauthorizedAccessException("Doctor profile not found.");

            var visit = new PatientVisit
            {
                PatientId = dto.PatientId,
                DoctorId = doctor.Id,
                AppointmentId = dto.AppointmentId,
                ChiefComplaint = dto.ChiefComplaint,
                PresentIllnessHistory = dto.PresentIllnessHistory,
                Status = "active",
                VisitDate = DateOnly.FromDateTime(DateTime.UtcNow),
                CreatedAt = DateTime.UtcNow,
            };

            await _unitOfWork.Repository<PatientVisit>().AddAsync(visit);
            await _unitOfWork.SaveChangesAsync();

            return MapVisit(visit);
        }

        public async Task<PatientVisitDto?> GetVisitAsync(int visitId)
        {
            var visit = await _unitOfWork.Repository<PatientVisit>().GetByIdAsync(visitId);
            return visit == null ? null : MapVisit(visit);
        }

        public async Task<PatientVisitDto?> UpdateVisitAsync(int doctorUserId, int visitId, UpdateVisitDto dto)
        {
            var doctor = await GetDoctorByUserIdAsync(doctorUserId) ?? throw new UnauthorizedAccessException("Doctor profile not found.");
            var visit = await _unitOfWork.Repository<PatientVisit>().GetByIdAsync(visitId);
            if (visit == null) return null;
            if (visit.DoctorId != doctor.Id) throw new UnauthorizedAccessException("Not allowed.");

            if (dto.ChiefComplaint != null) visit.ChiefComplaint = dto.ChiefComplaint;
            if (dto.PresentIllnessHistory != null) visit.PresentIllnessHistory = dto.PresentIllnessHistory;
            if (dto.ExaminationFindings != null) visit.ExaminationFindings = dto.ExaminationFindings;
            if (dto.Assessment != null) visit.Assessment = dto.Assessment;
            if (dto.Plan != null) visit.Plan = dto.Plan;
            if (dto.Notes != null) visit.Notes = dto.Notes;

            _unitOfWork.Repository<PatientVisit>().Update(visit);
            await _unitOfWork.SaveChangesAsync();

            return MapVisit(visit);
        }

        public async Task<bool> CloseVisitAsync(int doctorUserId, int visitId)
        {
            var doctor = await GetDoctorByUserIdAsync(doctorUserId) ?? throw new UnauthorizedAccessException("Doctor profile not found.");
            var visit = await _unitOfWork.Repository<PatientVisit>().GetByIdAsync(visitId);
            if (visit == null) return false;
            if (visit.DoctorId != doctor.Id) throw new UnauthorizedAccessException("Not allowed.");

            visit.Status = "closed";
            visit.ClosedAt = DateTime.UtcNow;
            _unitOfWork.Repository<PatientVisit>().Update(visit);
            await _unitOfWork.SaveChangesAsync();
            return true;
        }

        public async Task<IEnumerable<PatientVisitDto>> GetVisitsForPatientAsync(int patientId)
        {
            var items = await _unitOfWork.Repository<PatientVisit>().FindAsync(v => v.PatientId == patientId);
            return items.OrderByDescending(v => v.CreatedAt).Select(MapVisit).ToList();
        }

        public async Task<IEnumerable<PatientVisitDto>> GetTodayVisitsForDoctorAsync(int doctorUserId)
        {
            var doctor = await GetDoctorByUserIdAsync(doctorUserId) ?? throw new UnauthorizedAccessException("Doctor profile not found.");
            var today = DateOnly.FromDateTime(DateTime.UtcNow);
            var items = await _unitOfWork.Repository<PatientVisit>().FindAsync(v => v.DoctorId == doctor.Id && v.VisitDate == today);
            return items.OrderByDescending(v => v.CreatedAt).Select(MapVisit).ToList();
        }

        public async Task<VisitSummaryDto?> GetVisitSummaryAsync(int doctorUserId, int visitId)
        {
            var doctor = await GetDoctorByUserIdAsync(doctorUserId) ?? throw new UnauthorizedAccessException("Doctor profile not found.");
            var visit = await _unitOfWork.Repository<PatientVisit>().GetByIdAsync(visitId);
            if (visit == null) return null;
            if (visit.DoctorId != doctor.Id) throw new UnauthorizedAccessException("Not allowed.");

            return new VisitSummaryDto(visit.Id, visit.PatientId, visit.DoctorId, visit.VisitDate, visit.ChiefComplaint, visit.SummarySnapshot);
        }

        // Symptoms
        public async Task<VisitSymptomDto> AddSymptomAsync(int doctorOrNurseUserId, int visitId, CreateVisitSymptomDto dto)
        {
            var visit = await _unitOfWork.Repository<PatientVisit>().GetByIdAsync(visitId)
                ?? throw new InvalidOperationException("Visit not found.");

            var symptom = new Symptom
            {
                PatientVisitId = visitId,
                Name = dto.Name,
                Severity = dto.Severity,
                Duration = dto.Duration,
                Onset = dto.Onset,
                Progression = dto.Progression,
                Location = dto.Location,
                IsChronic = dto.IsChronic,
                Notes = dto.Notes,
                CreatedAt = DateTime.UtcNow,
            };

            await _unitOfWork.Repository<Symptom>().AddAsync(symptom);
            await _unitOfWork.SaveChangesAsync();
            return MapSymptom(symptom);
        }

        public async Task<IEnumerable<VisitSymptomDto>> GetSymptomsAsync(int visitId)
        {
            var items = await _unitOfWork.Repository<Symptom>().FindAsync(s => s.PatientVisitId == visitId);
            return items.OrderByDescending(s => s.CreatedAt).Select(MapSymptom).ToList();
        }

        public async Task<bool> DeleteSymptomAsync(int doctorUserId, int symptomId)
        {
            var doctor = await GetDoctorByUserIdAsync(doctorUserId) ?? throw new UnauthorizedAccessException("Doctor profile not found.");
            var symptom = await _unitOfWork.Repository<Symptom>().GetByIdAsync(symptomId);
            if (symptom == null) return false;

            var visit = await _unitOfWork.Repository<PatientVisit>().GetByIdAsync(symptom.PatientVisitId);
            if (visit == null || visit.DoctorId != doctor.Id) throw new UnauthorizedAccessException("Not allowed.");

            _unitOfWork.Repository<Symptom>().Delete(symptom);
            await _unitOfWork.SaveChangesAsync();
            return true;
        }

        public async Task<IEnumerable<VisitSymptomDto>> GetSymptomHistoryForPatientAsync(int doctorUserId, int patientId)
        {
            _ = await GetDoctorByUserIdAsync(doctorUserId) ?? throw new UnauthorizedAccessException("Doctor profile not found.");

            // all visits for patient
            var visits = await _unitOfWork.Repository<PatientVisit>().FindAsync(v => v.PatientId == patientId);
            var visitIds = visits.Select(v => v.Id).ToHashSet();

            var symptoms = await _unitOfWork.Repository<Symptom>().FindAsync(s => visitIds.Contains(s.PatientVisitId));
            return symptoms.OrderByDescending(s => s.CreatedAt).Select(MapSymptom).ToList();
        }

        // Clinical vitals
        public async Task<VisitVitalDto> AddVisitVitalAsync(int doctorOrNurseUserId, int visitId, CreateVisitVitalDto dto)
        {
            var visit = await _unitOfWork.Repository<PatientVisit>().GetByIdAsync(visitId)
                ?? throw new InvalidOperationException("Visit not found.");

            var user = await _unitOfWork.Repository<User>().GetByIdAsync(doctorOrNurseUserId);
            var role = user?.Role ?? "";
            var recordedBy = role.Equals("Nurse", StringComparison.OrdinalIgnoreCase) ? "nurse" : "doctor";

            var vital = new VisitVitalSign
            {
                PatientId = visit.PatientId,
                PatientVisitId = visitId,
                Type = dto.Type,
                Value = dto.Value,
                Value2 = dto.Value2,
                Unit = dto.Unit,
                IsAbnormal = dto.IsAbnormal,
                NormalRangeMin = dto.NormalRangeMin,
                NormalRangeMax = dto.NormalRangeMax,
                RecordedBy = recordedBy,
                Notes = dto.Notes,
                RecordedAt = DateTime.UtcNow,
            };

            await _unitOfWork.Repository<VisitVitalSign>().AddAsync(vital);
            await _unitOfWork.SaveChangesAsync();
            return MapVital(vital);
        }

        public async Task<IEnumerable<VisitVitalDto>> GetVisitVitalsAsync(int visitId)
        {
            var items = await _unitOfWork.Repository<VisitVitalSign>().FindAsync(v => v.PatientVisitId == visitId);
            return items.OrderByDescending(v => v.RecordedAt).Select(MapVital).ToList();
        }

        public async Task<bool> DeleteClinicalVitalAsync(int vitalId)
        {
            var item = await _unitOfWork.Repository<VisitVitalSign>().GetByIdAsync(vitalId);
            if (item == null) return false;
            _unitOfWork.Repository<VisitVitalSign>().Delete(item);
            await _unitOfWork.SaveChangesAsync();
            return true;
        }
    }
}
