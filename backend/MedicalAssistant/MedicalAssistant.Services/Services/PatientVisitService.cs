using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.DoctorsModule;
using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Domain.Entities.UserModule;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.PatientVisits;
using Microsoft.AspNetCore.Http;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MedicalAssistant.Services.Services
{
    public class PatientVisitService : IPatientVisitService
    {
        private readonly IUnitOfWork _unitOfWork;
        private readonly IPhotoService _photoService;

        public PatientVisitService(IUnitOfWork unitOfWork, IPhotoService photoService)
        {
            _unitOfWork = unitOfWork;
            _photoService = photoService;
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

        private static VisitPrescriptionDto MapPrescription(VisitPrescription p) => new(
            p.Id,
            p.PatientVisitId,
            p.MedicationName,
            p.GenericName,
            p.Dosage,
            p.Form,
            p.Frequency,
            p.TimesPerDay,
            p.SpecificTimes,
            p.Duration,
            p.Quantity,
            p.Instructions,
            p.IsChronic,
            p.Refills,
            p.IsDispensed,
            p.Notes,
            p.CreatedAt
        );

        private static VisitDocumentDto MapDocument(VisitDocument d) => new(
            d.Id,
            d.PatientVisitId,
            d.DocumentType,
            d.Title,
            d.FileUrl,
            d.FileType,
            d.Description,
            d.UploadedAt
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
            if (dto.FollowUpRequired.HasValue) visit.FollowUpRequired = dto.FollowUpRequired.Value;
            if (dto.FollowUpAfterDays.HasValue) visit.FollowUpAfterDays = dto.FollowUpAfterDays.Value;
            if (dto.FollowUpNotes != null) visit.FollowUpNotes = dto.FollowUpNotes;

            // Update Symptoms
            if (dto.Symptoms != null)
            {
                var existingSymptoms = await _unitOfWork.Repository<Symptom>().FindAsync(s => s.PatientVisitId == visitId);
                foreach (var s in existingSymptoms) _unitOfWork.Repository<Symptom>().Delete(s);
                foreach (var s in dto.Symptoms)
                {
                    await _unitOfWork.Repository<Symptom>().AddAsync(new Symptom
                    {
                        PatientVisitId = visitId,
                        Name = s.Name,
                        Severity = s.Severity,
                        Onset = s.Onset,
                        Location = s.Location,
                        Duration = s.Duration,
                        IsChronic = s.IsChronic,
                        CreatedAt = DateTime.UtcNow,
                    });
                }
            }

            // Update Prescriptions
            if (dto.Prescriptions != null)
            {
                var existingPrescriptions = await _unitOfWork.Repository<VisitPrescription>().FindAsync(p => p.PatientVisitId == visitId);
                foreach (var p in existingPrescriptions) _unitOfWork.Repository<VisitPrescription>().Delete(p);
                foreach (var p in dto.Prescriptions)
                {
                    await _unitOfWork.Repository<VisitPrescription>().AddAsync(new VisitPrescription
                    {
                        PatientVisitId = visitId,
                        MedicationName = p.MedicationName,
                        Dosage = p.Dosage,
                        Frequency = p.Frequency,
                        Duration = p.Duration,
                        Quantity = p.Quantity,
                        Instructions = p.Instructions,
                        IsChronic = p.IsChronic,
                        CreatedAt = DateTime.UtcNow,
                    });
                }
            }

            // Update Vitals
            if (dto.VitalSigns != null)
            {
                var existingVitals = await _unitOfWork.Repository<VisitVitalSign>().FindAsync(v => v.PatientVisitId == visitId);
                foreach (var v in existingVitals) _unitOfWork.Repository<VisitVitalSign>().Delete(v);
                foreach (var v in dto.VitalSigns)
                {
                    await _unitOfWork.Repository<VisitVitalSign>().AddAsync(new VisitVitalSign
                    {
                        PatientId = visit.PatientId,
                        PatientVisitId = visitId,
                        Type = v.Type,
                        Value = v.Value,
                        Value2 = v.Value2,
                        Unit = v.Unit,
                        IsAbnormal = v.IsAbnormal,
                        RecordedBy = "doctor",
                        RecordedAt = DateTime.UtcNow,
                    });
                }
            }

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

            var patient = visit.Patient ?? await _unitOfWork.Repository<Patient>().GetByIdAsync(visit.PatientId);
            var age = patient != null && patient.DateOfBirth > DateTime.MinValue
                ? DateTime.UtcNow.Year - patient.DateOfBirth.Year
                : 0;

            var allergies = patient?.AllergyRecords?.Select(a => new AllergySummaryDto(
                a.AllergenName,
                a.Severity,
                a.ReactionDescription ?? string.Empty
            )).ToList() ?? new List<AllergySummaryDto>();

            var symptoms = visit.Symptoms?.Select(s => new SymptomSummaryDto(
                s.Name,
                s.Severity,
                s.Onset ?? string.Empty,
                s.Location,
                s.Duration,
                s.IsChronic
            )).ToList() ?? new List<SymptomSummaryDto>();

            var vitals = visit.VitalSigns?.Select(v => new VitalSummaryDto(
                v.Type,
                v.Value,
                v.Value2,
                v.Unit,
                v.IsAbnormal
            )).ToList() ?? new List<VitalSummaryDto>();

            var prescriptions = visit.Prescriptions?.Select(p => new PrescriptionSummaryDto(
                p.MedicationName,
                p.Dosage,
                p.Frequency,
                p.Duration,
                p.Quantity,
                p.Instructions,
                p.IsChronic
            )).ToList() ?? new List<PrescriptionSummaryDto>();

            return new VisitSummaryDto(
                visit.Id,
                patient?.FullName ?? string.Empty,
                age,
                patient?.BloodType ?? string.Empty,
                allergies,
                visit.VisitDate,
                visit.ChiefComplaint,
                visit.ExaminationFindings,
                visit.Assessment,
                visit.Plan,
                vitals,
                prescriptions,
                symptoms,
                visit.Notes,
                visit.FollowUpRequired,
                visit.FollowUpAfterDays,
                visit.FollowUpNotes
            );
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

        // Prescriptions
        public async Task<VisitPrescriptionDto> AddPrescriptionAsync(int doctorUserId, int visitId, CreateVisitPrescriptionDto dto)
        {
            var doctor = await GetDoctorByUserIdAsync(doctorUserId) ?? throw new UnauthorizedAccessException("Doctor profile not found.");
            var visit = await _unitOfWork.Repository<PatientVisit>().GetByIdAsync(visitId)
                ?? throw new InvalidOperationException("Visit not found.");
            if (visit.DoctorId != doctor.Id) throw new UnauthorizedAccessException("Not allowed.");

            var line = new VisitPrescription
            {
                PatientVisitId = visitId,
                MedicationName = dto.MedicationName,
                GenericName = dto.GenericName,
                Dosage = dto.Dosage,
                Form = dto.Form,
                Frequency = dto.Frequency,
                TimesPerDay = dto.TimesPerDay,
                SpecificTimes = dto.SpecificTimes,
                Duration = dto.Duration,
                Quantity = dto.Quantity,
                Instructions = dto.Instructions,
                IsChronic = dto.IsChronic,
                Refills = dto.Refills,
                Notes = dto.Notes,
                IsDispensed = false,
                CreatedAt = DateTime.UtcNow,
            };

            await _unitOfWork.Repository<VisitPrescription>().AddAsync(line);
            await _unitOfWork.SaveChangesAsync();
            return MapPrescription(line);
        }

        public async Task<IEnumerable<VisitPrescriptionDto>> GetPrescriptionsAsync(int visitId)
        {
            var items = await _unitOfWork.Repository<VisitPrescription>().FindAsync(p => p.PatientVisitId == visitId);
            return items.OrderByDescending(p => p.CreatedAt).Select(MapPrescription).ToList();
        }

        public async Task<VisitPrescriptionDto?> UpdatePrescriptionAsync(int doctorUserId, int prescriptionId, UpdateVisitPrescriptionDto dto)
        {
            var doctor = await GetDoctorByUserIdAsync(doctorUserId) ?? throw new UnauthorizedAccessException("Doctor profile not found.");
            var line = await _unitOfWork.Repository<VisitPrescription>().GetByIdAsync(prescriptionId);
            if (line == null) return null;

            var visit = await _unitOfWork.Repository<PatientVisit>().GetByIdAsync(line.PatientVisitId);
            if (visit == null || visit.DoctorId != doctor.Id) throw new UnauthorizedAccessException("Not allowed.");

            if (dto.MedicationName != null) line.MedicationName = dto.MedicationName;
            if (dto.GenericName != null) line.GenericName = dto.GenericName;
            if (dto.Dosage != null) line.Dosage = dto.Dosage;
            if (dto.Form != null) line.Form = dto.Form;
            if (dto.Frequency != null) line.Frequency = dto.Frequency;
            if (dto.TimesPerDay.HasValue) line.TimesPerDay = dto.TimesPerDay.Value;
            if (dto.SpecificTimes != null) line.SpecificTimes = dto.SpecificTimes;
            if (dto.Duration != null) line.Duration = dto.Duration;
            if (dto.Quantity.HasValue) line.Quantity = dto.Quantity;
            if (dto.Instructions != null) line.Instructions = dto.Instructions;
            if (dto.IsChronic.HasValue) line.IsChronic = dto.IsChronic.Value;
            if (dto.Refills.HasValue) line.Refills = dto.Refills.Value;
            if (dto.Notes != null) line.Notes = dto.Notes;

            _unitOfWork.Repository<VisitPrescription>().Update(line);
            await _unitOfWork.SaveChangesAsync();
            return MapPrescription(line);
        }

        public async Task<bool> DeletePrescriptionAsync(int doctorUserId, int prescriptionId)
        {
            var doctor = await GetDoctorByUserIdAsync(doctorUserId) ?? throw new UnauthorizedAccessException("Doctor profile not found.");
            var line = await _unitOfWork.Repository<VisitPrescription>().GetByIdAsync(prescriptionId);
            if (line == null) return false;

            var visit = await _unitOfWork.Repository<PatientVisit>().GetByIdAsync(line.PatientVisitId);
            if (visit == null || visit.DoctorId != doctor.Id) throw new UnauthorizedAccessException("Not allowed.");

            _unitOfWork.Repository<VisitPrescription>().Delete(line);
            await _unitOfWork.SaveChangesAsync();
            return true;
        }

        public async Task<bool> DispensePrescriptionAsync(int pharmacistUserId, int prescriptionId)
        {
            var user = await _unitOfWork.Repository<User>().GetByIdAsync(pharmacistUserId);
            if (user == null || !string.Equals(user.Role, "Pharmacist", StringComparison.OrdinalIgnoreCase))
                throw new UnauthorizedAccessException("Not allowed.");

            var line = await _unitOfWork.Repository<VisitPrescription>().GetByIdAsync(prescriptionId);
            if (line == null) return false;

            if (!line.IsDispensed)
            {
                line.IsDispensed = true;
                _unitOfWork.Repository<VisitPrescription>().Update(line);
                await _unitOfWork.SaveChangesAsync();
            }

            return true;
        }

        public async Task<PatientHistoryDto?> GetPatientHistoryAsync(int doctorUserId, int patientId)
        {
            var doctor = await GetDoctorByUserIdAsync(doctorUserId) ?? throw new UnauthorizedAccessException("Doctor profile not found.");
            
            var patient = await _unitOfWork.Repository<Patient>().GetByIdAsync(patientId);
            if (patient == null) return null;

            var visits = await _unitOfWork.Repository<PatientVisit>().FindAsync(v => v.PatientId == patientId);
            var lastVisits = visits.OrderByDescending(v => v.CreatedAt).Take(5).Select(v => new LastVisitSummaryDto(
                v.Id.ToString(),
                v.VisitDate.ToString("yyyy-MM-dd"),
                v.ChiefComplaint
            )).ToList();

            var allergies = patient.AllergyRecords?.Select(a => new AllergySummaryDto(
                a.AllergenName,
                a.Severity,
                a.ReactionDescription ?? string.Empty
            )).ToList() ?? new List<AllergySummaryDto>();

            var chronicDiseases = patient.ChronicDiseaseMonitors?.Select(c => new ChronicDiseaseSummaryDto(
                c.Id.ToString(),
                c.DiseaseName,
                c.TargetValues
            )).ToList() ?? new List<ChronicDiseaseSummaryDto>();

            var medications = patient.MedicationTrackers?.Select(m => new MedicationSummaryDto(
                m.Id.ToString(),
                m.MedicationName,
                m.Dosage,
                m.Form
            )).ToList() ?? new List<MedicationSummaryDto>();

            var latestVitals = new Dictionary<string, string>();
            var lastVisitWithVitals = visits.OrderByDescending(v => v.CreatedAt).FirstOrDefault(v => v.VitalSigns.Any());
            if (lastVisitWithVitals != null)
            {
                foreach (var vital in lastVisitWithVitals.VitalSigns)
                {
                    var valueStr = vital.Value2.HasValue ? $"{vital.Value}/{vital.Value2.Value}" : vital.Value.ToString();
                    latestVitals[vital.Type] = valueStr;
                }
            }

            return new PatientHistoryDto(
                patient.BloodType ?? string.Empty,
                allergies,
                chronicDiseases,
                medications,
                latestVitals,
                lastVisits
            );
        }

        // Documents
        public async Task<VisitDocumentDto> UploadDocumentAsync(int uploaderUserId, int visitId, IFormFile file, string documentType, string title, string? description)
        {
            var user = await _unitOfWork.Repository<User>().GetByIdAsync(uploaderUserId)
                ?? throw new UnauthorizedAccessException("Invalid token.");

            if (!string.Equals(user.Role, "Doctor", StringComparison.OrdinalIgnoreCase) &&
                !string.Equals(user.Role, "Nurse", StringComparison.OrdinalIgnoreCase))
            {
                throw new UnauthorizedAccessException("Not allowed.");
            }

            var visit = await _unitOfWork.Repository<PatientVisit>().GetByIdAsync(visitId)
                ?? throw new InvalidOperationException("Visit not found.");

            if (string.Equals(user.Role, "Doctor", StringComparison.OrdinalIgnoreCase))
            {
                var doctor = await GetDoctorByUserIdAsync(uploaderUserId) ?? throw new UnauthorizedAccessException("Doctor profile not found.");
                if (visit.DoctorId != doctor.Id) throw new UnauthorizedAccessException("Not allowed.");
            }

            if (file == null || file.Length == 0) throw new InvalidOperationException("File is required.");

            var url = await _photoService.UploadFileAsync(file);

            var doc = new VisitDocument
            {
                PatientVisitId = visitId,
                DocumentType = documentType ?? string.Empty,
                Title = string.IsNullOrWhiteSpace(title) ? file.FileName : title,
                FileUrl = url,
                FileType = file.ContentType ?? string.Empty,
                Description = description,
                UploadedAt = DateTime.UtcNow,
            };

            await _unitOfWork.Repository<VisitDocument>().AddAsync(doc);
            await _unitOfWork.SaveChangesAsync();
            return MapDocument(doc);
        }

        public async Task<IEnumerable<VisitDocumentDto>> GetDocumentsAsync(int visitId)
        {
            var items = await _unitOfWork.Repository<VisitDocument>().FindAsync(d => d.PatientVisitId == visitId);
            return items.OrderByDescending(d => d.UploadedAt).Select(MapDocument).ToList();
        }

        public async Task<bool> DeleteDocumentAsync(int requesterUserId, int documentId)
        {
            var user = await _unitOfWork.Repository<User>().GetByIdAsync(requesterUserId)
                ?? throw new UnauthorizedAccessException("Invalid token.");

            var doc = await _unitOfWork.Repository<VisitDocument>().GetByIdAsync(documentId);
            if (doc == null) return false;

            if (string.Equals(user.Role, "Admin", StringComparison.OrdinalIgnoreCase))
            {
                _unitOfWork.Repository<VisitDocument>().Delete(doc);
                await _unitOfWork.SaveChangesAsync();
                return true;
            }

            if (string.Equals(user.Role, "Doctor", StringComparison.OrdinalIgnoreCase))
            {
                var doctor = await GetDoctorByUserIdAsync(requesterUserId) ?? throw new UnauthorizedAccessException("Doctor profile not found.");
                var visit = await _unitOfWork.Repository<PatientVisit>().GetByIdAsync(doc.PatientVisitId);
                if (visit == null || visit.DoctorId != doctor.Id) throw new UnauthorizedAccessException("Not allowed.");

                _unitOfWork.Repository<VisitDocument>().Delete(doc);
                await _unitOfWork.SaveChangesAsync();
                return true;
            }

            throw new UnauthorizedAccessException("Not allowed.");
        }
    }
}
